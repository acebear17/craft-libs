//! The UV-space G-buffer: which triangle (and where on it) each texel of a texture set maps
//! to. Painting, compositing, baking and padding all work from this one buffer, so they agree
//! on which texel is which surface point.

use glam::{Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::{GeomError, Mesh, Result};

/// Texel not covered by any triangle.
pub const NO_TRI: u32 = u32::MAX;

const BAND: u32 = 32;

/// Largest width or height of a G-buffer, in texels.
pub const MAX_SIDE: u32 = 16_384;

/// Largest texel count of a G-buffer (8,192 × 8,192). Each texel takes 12 bytes, so a full
/// buffer takes 768 MiB and its largest vector (`bary`) 512 MiB, which fits a 32-bit address
/// space.
pub const MAX_TEXELS: usize = 1 << 26;

/// Largest number of (triangle, band of 32 rows) pairs a G-buffer bins before rasterizing: a
/// triangle counts once for every band it touches. The bins take 4 bytes per entry, 512 MiB at
/// most.
pub const MAX_BAND_ENTRIES: usize = 1 << 27;

#[derive(Clone, Debug, PartialEq)]
pub struct UvGBuffer {
    pub width: u32,
    pub height: u32,
    /// Triangle under each texel center, [`NO_TRI`] when uncovered.
    pub tri: Vec<u32>,
    /// Barycentric weights of the triangle's second and third vertices.
    pub bary: Vec<[f32; 2]>,
    /// Texels written by more than one triangle (overlapping UVs).
    pub overlaps: u32,
}

/// Interpolated surface attributes at a texel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub pos: Vec3,
    /// Interpolated shading normal (normalized).
    pub nrm: Vec3,
    /// Flat triangle normal (for ray offsets).
    pub geo_nrm: Vec3,
    pub tangent: Vec4,
    pub uv: Vec2,
    pub tri: u32,
}

/// Checks a requested G-buffer size against [`MAX_SIDE`] and [`MAX_TEXELS`] and returns its
/// texel count.
fn checked_texels(width: u32, height: u32) -> Result<usize> {
    for (name, side) in [("width", width), ("height", height)] {
        if side == 0 || side > MAX_SIDE {
            return Err(GeomError::Invalid(format!("G-buffer {name} {side} is outside 1..={MAX_SIDE}")));
        }
    }
    // Both sides are at most MAX_SIDE, so the product fits in usize on 32-bit targets too.
    let n = width as usize * height as usize;
    if n > MAX_TEXELS {
        return Err(GeomError::Invalid(format!("G-buffer {width}×{height} has more than {MAX_TEXELS} texels")));
    }
    Ok(n)
}

/// A vector of `n` copies of `value`, or an error when the memory cannot be reserved.
fn filled<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|e| GeomError::Invalid(format!("G-buffer of {n} texels: {e}")))?;
    v.resize(n, value);
    Ok(v)
}

fn edge(a: Vec2, b: Vec2, p: Vec2) -> f32 {
    (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
}

impl UvGBuffer {
    /// Rasterize the triangles of material `material` at `width × height`.
    ///
    /// Returns an error when `width` or `height` is 0 or above [`MAX_SIDE`], when
    /// `width × height` is above [`MAX_TEXELS`], when the triangles touch more than
    /// [`MAX_BAND_ENTRIES`] bands of 32 rows in all, or when the memory cannot be reserved.
    pub fn new(mesh: &Mesh, material: usize, width: u32, height: u32) -> Result<UvGBuffer> {
        let n = checked_texels(width, height)?;
        let (w, h) = (width, height);
        let size = Vec2::new(w as f32, h as f32);
        let bands = h.div_ceil(BAND) as usize;
        let span = |t: usize| -> Option<(usize, usize)> {
            let [a, b, c] = mesh.tri(t)?;
            let (ua, ub, uc) = (mesh.uvs.get(a)?, mesh.uvs.get(b)?, mesh.uvs.get(c)?);
            let y0 = (ua.y.min(ub.y).min(uc.y) * size.y).floor().max(0.0) as i64;
            let y1 = (ua.y.max(ub.y).max(uc.y) * size.y).ceil().min(size.y) as i64;
            if y1 <= y0 || !ua.is_finite() || !ub.is_finite() || !uc.is_finite() {
                return None;
            }
            Some(((y0 / BAND as i64) as usize, ((y1 - 1).max(0) / BAND as i64) as usize))
        };
        // The triangles of each band, in mesh order, as one list: band `i` holds
        // `bins[start[i]..start[i + 1]]`. A counting pass sizes the list before it is reserved.
        let mut start = vec![0usize; bands + 1];
        for t in mesh.triangles_of(material) {
            let Some((b0, b1)) = span(t) else { continue };
            for s in start.iter_mut().skip(b0 + 1).take(b1 + 1 - b0) {
                *s += 1;
            }
        }
        for i in 1..start.len() {
            start[i] = start[i].saturating_add(start[i - 1]);
        }
        let entries = start.last().copied().unwrap_or(0);
        if entries > MAX_BAND_ENTRIES {
            return Err(GeomError::Invalid(format!("G-buffer triangles touch {entries} bands in all, more than {MAX_BAND_ENTRIES}")));
        }
        let mut bins = filled(entries, 0u32)?;
        let mut next = start.clone();
        for t in mesh.triangles_of(material) {
            let Some((b0, b1)) = span(t) else { continue };
            for k in next.iter_mut().take(b1 + 1).skip(b0) {
                if let Some(slot) = bins.get_mut(*k) {
                    *slot = t as u32;
                }
                *k += 1;
            }
        }
        let mut tri = filled(n, NO_TRI)?;
        let mut bary = filled(n, [0.0f32; 2])?;
        let band_len = BAND as usize * w as usize;
        let overlaps: u32 = tri
            .par_chunks_mut(band_len)
            .zip(bary.par_chunks_mut(band_len))
            .enumerate()
            .map(|(bi, (tri, bary))| {
                let mut overlaps = 0u32;
                let row0 = bi as u32 * BAND;
                let rows = (tri.len() as u32) / w;
                let band = match (start.get(bi), start.get(bi + 1)) {
                    (Some(&s), Some(&e)) => bins.get(s..e).unwrap_or(&[]),
                    _ => &[],
                };
                for &t in band {
                    let Some([a, b, c]) = mesh.tri(t as usize) else { continue };
                    let (Some(&ua), Some(&ub), Some(&uc)) = (mesh.uvs.get(a), mesh.uvs.get(b), mesh.uvs.get(c)) else { continue };
                    let (pa, pb, pc) = (ua * size, ub * size, uc * size);
                    let area = edge(pa, pb, pc);
                    if area.abs() < 1e-12 {
                        continue;
                    }
                    let eps = -1e-6 * area.abs();
                    let x0 = pa.x.min(pb.x).min(pc.x).floor().max(0.0) as u32;
                    let x1 = (pa.x.max(pb.x).max(pc.x).ceil().min(w as f32) as u32).min(w);
                    let y0 = (pa.y.min(pb.y).min(pc.y).floor().max(row0 as f32) as u32).max(row0);
                    let y1 = (pa.y.max(pb.y).max(pc.y).ceil() as u32).min(row0 + rows);
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                            let (w0, w1, w2) = (edge(pb, pc, p) * area.signum(), edge(pc, pa, p) * area.signum(), edge(pa, pb, p) * area.signum());
                            if w0 < eps || w1 < eps || w2 < eps {
                                continue;
                            }
                            let i = (y - row0) as usize * w as usize + x as usize;
                            if let (Some(slot), Some(bw)) = (tri.get_mut(i), bary.get_mut(i)) {
                                if *slot != NO_TRI {
                                    overlaps += 1;
                                }
                                *slot = t;
                                let inv = 1.0 / area.abs();
                                *bw = [w1 * inv, w2 * inv];
                            }
                        }
                    }
                }
                overlaps
            })
            .sum();
        Ok(UvGBuffer { width: w, height: h, tri, bary, overlaps })
    }

    pub fn covered(&self, i: usize) -> bool {
        self.tri.get(i).is_some_and(|t| *t != NO_TRI)
    }

    pub fn coverage_mask(&self) -> Vec<bool> {
        self.tri.iter().map(|t| *t != NO_TRI).collect()
    }

    pub fn covered_count(&self) -> usize {
        self.tri.iter().filter(|t| **t != NO_TRI).count()
    }

    /// Surface attributes at texel index `i` (None when uncovered).
    pub fn surface(&self, mesh: &Mesh, i: usize) -> Option<Surface> {
        let t = *self.tri.get(i)?;
        if t == NO_TRI {
            return None;
        }
        let [b1, b2] = *self.bary.get(i)?;
        surface_at(mesh, t, b1, b2)
    }
}

/// Interpolate triangle `t` at barycentrics (weights of vertices 1 and 2).
pub fn surface_at(mesh: &Mesh, t: u32, b1: f32, b2: f32) -> Option<Surface> {
    let [a, b, c] = mesh.tri(t as usize)?;
    let b0 = 1.0 - b1 - b2;
    let p = |v: &[Vec3]| -> Option<Vec3> { Some(*v.get(a)? * b0 + *v.get(b)? * b1 + *v.get(c)? * b2) };
    let pos = p(&mesh.positions)?;
    let nrm = p(&mesh.normals)?.try_normalize().unwrap_or(Vec3::Y);
    let (pa, pb, pc) = (*mesh.positions.get(a)?, *mesh.positions.get(b)?, *mesh.positions.get(c)?);
    let geo_nrm = (pb - pa).cross(pc - pa).try_normalize().unwrap_or(nrm);
    let tan = *mesh.tangents.get(a)? * b0 + *mesh.tangents.get(b)? * b1 + *mesh.tangents.get(c)? * b2;
    let uv = *mesh.uvs.get(a)? * b0 + *mesh.uvs.get(b)? * b1 + *mesh.uvs.get(c)? * b2;
    let tdir = tan.truncate().try_normalize().unwrap_or_else(|| nrm.any_orthonormal_vector());
    let w = mesh.tangents.get(a).map(|t| if t.w < 0.0 { -1.0 } else { 1.0 }).unwrap_or(1.0);
    Some(Surface { pos, nrm, geo_nrm, tangent: tdir.extend(w), uv, tri: t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn half_triangle_covers_texels_with_centers_inside() {
        // One triangle covering the lower-left half of UV space (v down: y = v·H).
        let m = Mesh::from_parts(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            None,
            vec![Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0)],
            vec![0, 1, 2],
            "m",
        )
        .unwrap();
        let g = UvGBuffer::new(&m, 0, 8, 8).unwrap();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let inside = (x as f32 + 0.5) <= (y as f32 + 0.5);
                assert_eq!(g.covered((y * 8 + x) as usize), inside, "({x},{y})");
            }
        }
        assert_eq!(g.overlaps, 0);
    }

    #[test]
    fn overlapping_and_degenerate_triangles() {
        let uv = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(0.5, 0.5)];
        let m = Mesh::from_parts(vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z], None, uv, vec![0, 1, 2, 0, 1, 2, 0, 3, 3], "m").unwrap();
        let g = UvGBuffer::new(&m, 0, 16, 16).unwrap();
        assert!(g.overlaps > 0);
        assert!(g.covered_count() > 0);
    }

    #[test]
    fn sizes_outside_the_limits_are_rejected() {
        // 2^27 + 1 texels wide: 32 rows of it overflow u32.
        assert!(checked_texels((1 << 27) + 1, 1).is_err());
        assert!(checked_texels(u32::MAX, u32::MAX).is_err());
        assert!(checked_texels(0, 8).is_err());
        assert!(checked_texels(8, 0).is_err());
        assert!(checked_texels(MAX_SIDE + 1, 1).is_err());
        assert!(checked_texels(1, MAX_SIDE + 1).is_err());
        // Both sides in range, but more than MAX_TEXELS texels.
        assert!(checked_texels(MAX_SIDE, MAX_SIDE).is_err());
        assert!(checked_texels(MAX_SIDE, 4_097).is_err());
        assert_eq!(checked_texels(MAX_SIDE, 4_096), Ok(MAX_TEXELS));
        assert_eq!(checked_texels(MAX_SIDE, 1), Ok(MAX_SIDE as usize));

        let m = primitives::sphere(1.0, 8, 4);
        assert!(UvGBuffer::new(&m, 0, MAX_SIDE + 1, 1).is_err());
        assert!(UvGBuffer::new(&m, 0, 0, 1).is_err());
        let g = UvGBuffer::new(&m, 0, 64, 32).unwrap();
        assert_eq!((g.width, g.height, g.tri.len(), g.bary.len()), (64, 32, 2_048, 2_048));
    }

    #[test]
    fn too_many_band_entries_are_rejected() {
        // 2^27 / 512 + 1 triangles, each spanning all 512 bands of a 1 × 16,384 buffer.
        let tall = (MAX_BAND_ENTRIES / 512 + 1) as u32;
        let mut m = Mesh {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            uvs: vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)],
            ..Mesh::default()
        };
        m.indices = (0..tall).flat_map(|_| [0, 1, 2]).collect();
        m.submeshes = vec![crate::SubMesh { name: "m".into(), material: 0, start: 0, count: tall * 3 }];
        assert!(UvGBuffer::new(&m, 0, 1, MAX_SIDE).is_err());
        m.indices.truncate(3);
        m.submeshes[0].count = 3;
        let g = UvGBuffer::new(&m, 0, 1, MAX_SIDE).unwrap();
        // The texel centers at u = 0.5 lie inside the triangle down to v = 0.5.
        assert_eq!(g.covered_count(), MAX_SIDE as usize / 2);
    }

    #[test]
    fn sphere_surface_lies_on_the_sphere() {
        let m = primitives::sphere(2.0, 64, 32);
        let g = UvGBuffer::new(&m, 0, 128, 128).unwrap();
        assert!(g.covered_count() > 128 * 128 * 9 / 10, "{}", g.covered_count());
        for i in (0..g.tri.len()).step_by(97) {
            if let Some(s) = g.surface(&m, i) {
                assert!((s.pos.length() - 2.0).abs() < 0.02);
                assert!(s.nrm.dot(s.pos.normalize()) > 0.99);
            }
        }
    }
}
