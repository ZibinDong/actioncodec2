
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel.hpp"

namespace py = pybind11;

namespace {

using Vec = std::vector<double>;

struct Quat {
  double x, y, z, w;
};

// ---- runtime ``_so3`` quaternion math (xyzw), NumPy evaluation order ----

Quat so3_normalize(const Quat& q) {
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm < 1e-12) throw std::invalid_argument("a pose contains a zero quaternion");
  return {q.x / norm, q.y / norm, q.z / norm, q.w / norm};
}

Quat so3_multiply(const Quat& a, const Quat& b) {
  const double cx = a.y * b.z - a.z * b.y;
  const double cy = a.z * b.x - a.x * b.z;
  const double cz = a.x * b.y - a.y * b.x;
  return so3_normalize({a.w * b.x + b.w * a.x + cx, a.w * b.y + b.w * a.y + cy,
                        a.w * b.z + b.w * a.z + cz,
                        a.w * b.w - (a.x * b.x + a.y * b.y + a.z * b.z)});
}

Quat so3_from_rotvec(double x, double y, double z) {
  const double angle = std::sqrt(x * x + y * y + z * z);
  const double scale = angle >= 1e-12 ? std::sin(0.5 * angle) / angle : 0.5;
  return so3_normalize({x * scale, y * scale, z * scale, std::cos(0.5 * angle)});
}

void so3_to_rotvec(const Quat& raw, double* out) {
  Quat q = so3_normalize(raw);
  if (q.w < 0.0) q = {-q.x, -q.y, -q.z, -q.w};
  const double vector_norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  const double angle = 2.0 * std::atan2(vector_norm, std::min(std::max(q.w, 0.0), 1.0));
  const double scale = vector_norm >= 1e-12 ? angle / vector_norm : 2.0;
  out[0] = q.x * scale;
  out[1] = q.y * scale;
  out[2] = q.z * scale;
}

// ---- scipy ``Rotation`` operations used by RotationSpline ----

Quat sp_from_quat(const Quat& q) {
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (!(norm > 0.0)) throw std::invalid_argument("found zero norm quaternions in `quat`");
  return {q.x / norm, q.y / norm, q.z / norm, q.w / norm};
}

Quat sp_inv(const Quat& q) { return {-q.x, -q.y, -q.z, q.w}; }

Quat sp_compose(const Quat& p, const Quat& q) {
  const double cx = p.y * q.z - p.z * q.y;
  const double cy = p.z * q.x - p.x * q.z;
  const double cz = p.x * q.y - p.y * q.x;
  const Quat r{p.w * q.x + q.w * p.x + cx, p.w * q.y + q.w * p.y + cy, p.w * q.z + q.w * p.z + cz,
               p.w * q.w - (p.x * q.x + p.y * q.y + p.z * q.z)};
  return sp_from_quat(r);
}

void sp_as_rotvec(Quat q, double* out) {
  if (q.w < 0.0) q = {-q.x, -q.y, -q.z, -q.w};
  const double angle = 2.0 * std::atan2(std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z), q.w);
  double scale;
  if (angle <= 1e-3) {
    const double angle2 = angle * angle;
    scale = 2.0 + angle2 / 12.0 + 7.0 * angle2 * angle2 / 2880.0;
  } else {
    scale = angle / std::sin(angle / 2.0);
  }
  out[0] = scale * q.x;
  out[1] = scale * q.y;
  out[2] = scale * q.z;
}

Quat sp_from_rotvec(const double* v) {
  const double angle = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  double scale;
  if (angle <= 1e-3) {
    const double angle2 = angle * angle;
    scale = 0.5 - angle2 / 48.0 + angle2 * angle2 / 3840.0;
  } else {
    scale = std::sin(angle / 2.0) / angle;
  }
  return {scale * v[0], scale * v[1], scale * v[2], std::cos(angle / 2.0)};
}

// ---- piecewise cubics (scipy PPoly with CubicHermiteSpline coefficients) ----

// PPoly interval: x[i] <= t < x[i+1], extrapolating with the end intervals.
int64_t ppoly_interval(const Vec& x, double t) {
  const int64_t n = static_cast<int64_t>(x.size());
  if (t >= x[n - 1]) return n - 2;
  if (t < x[0]) return 0;
  return static_cast<int64_t>(std::upper_bound(x.begin(), x.end(), t) - x.begin()) - 1;
}

// scipy ``evaluate_poly1`` for a cubic: sum of c[k-1-p] * s**p, powers built by
// repeated multiplication.
double ppoly_value(const double* c, double s) {
  double result = 0.0, z = 1.0;
  result = result + c[3] * z;
  z *= s;
  result = result + c[2] * z;
  z *= s;
  result = result + c[1] * z;
  z *= s;
  result = result + c[0] * z;
  return result;
}

// Hermite coefficients per interval and column: c[(k*4 + p)*cols + a] with
// p = 0..3 the descending powers, as CubicHermiteSpline stacks them.
Vec hermite_coefficients(const Vec& x, const Vec& y, const Vec& slopes, int64_t cols) {
  const int64_t n = static_cast<int64_t>(x.size());
  Vec c(static_cast<size_t>((n - 1) * 4 * cols));
  for (int64_t k = 0; k < n - 1; ++k) {
    const double dx = x[k + 1] - x[k];
    for (int64_t a = 0; a < cols; ++a) {
      const double y0 = y[k * cols + a], y1 = y[(k + 1) * cols + a];
      const double d0 = slopes[k * cols + a], d1 = slopes[(k + 1) * cols + a];
      const double slope = (y1 - y0) / dx;
      const double t = (d0 + d1 - 2.0 * slope) / dx;
      c[(k * 4 + 0) * cols + a] = t / dx;
      c[(k * 4 + 1) * cols + a] = (slope - d0) / dx - t;
      c[(k * 4 + 2) * cols + a] = d0;
      c[(k * 4 + 3) * cols + a] = y0;
    }
  }
  return c;
}

void ppoly_evaluate(const Vec& x, const Vec& c, int64_t cols, const Vec& times, int64_t out_cols,
                    int64_t col_offset, double* out) {
  for (size_t i = 0; i < times.size(); ++i) {
    const int64_t k = ppoly_interval(x, times[i]);
    const double s = times[i] - x[k];
    for (int64_t a = 0; a < cols; ++a) {
      const double coeff[4] = {c[(k * 4 + 0) * cols + a], c[(k * 4 + 1) * cols + a],
                               c[(k * 4 + 2) * cols + a], c[(k * 4 + 3) * cols + a]};
      out[i * out_cols + col_offset + a] = ppoly_value(coeff, s);
    }
  }
}

// LAPACK dgtsv (reference algorithm) for several right-hand sides b[n][cols].
void gtsv(Vec dl, Vec d, Vec du, Vec& b, int64_t cols) {
  const int64_t n = static_cast<int64_t>(d.size());
  auto pivot = [&](int64_t i, bool last) {
    if (std::fabs(d[i]) >= std::fabs(dl[i])) {
      if (d[i] == 0.0) throw std::invalid_argument("singular matrix");
      const double fact = dl[i] / d[i];
      d[i + 1] = d[i + 1] - fact * du[i];
      for (int64_t j = 0; j < cols; ++j) b[(i + 1) * cols + j] = b[(i + 1) * cols + j] - fact * b[i * cols + j];
      if (!last) dl[i] = 0.0;
    } else {
      const double fact = d[i] / dl[i];
      d[i] = dl[i];
      const double temp = d[i + 1];
      d[i + 1] = du[i] - fact * temp;
      if (!last) {
        dl[i] = du[i + 1];
        du[i + 1] = -fact * dl[i];
      }
      du[i] = temp;
      for (int64_t j = 0; j < cols; ++j) {
        const double swap = b[i * cols + j];
        b[i * cols + j] = b[(i + 1) * cols + j];
        b[(i + 1) * cols + j] = swap - fact * b[(i + 1) * cols + j];
      }
    }
  };
  for (int64_t i = 0; i + 2 < n; ++i) pivot(i, false);
  if (n > 1) pivot(n - 2, true);
  if (d[n - 1] == 0.0) throw std::invalid_argument("singular matrix");
  for (int64_t j = 0; j < cols; ++j) {
    b[(n - 1) * cols + j] = b[(n - 1) * cols + j] / d[n - 1];
    if (n > 1)
      b[(n - 2) * cols + j] = (b[(n - 2) * cols + j] - du[n - 2] * b[(n - 1) * cols + j]) / d[n - 2];
    for (int64_t i = n - 3; i >= 0; --i) {
      b[i * cols + j] =
          (b[i * cols + j] - du[i] * b[(i + 1) * cols + j] - dl[i] * b[(i + 2) * cols + j]) / d[i];
    }
  }
}

// scipy CubicSpline(bc_type="natural") slopes at the knots.
Vec natural_cubic_slopes(const Vec& x, const Vec& y, int64_t cols) {
  const int64_t n = static_cast<int64_t>(x.size());
  Vec dx(n - 1), slope((n - 1) * cols);
  for (int64_t k = 0; k < n - 1; ++k) {
    dx[k] = x[k + 1] - x[k];
    for (int64_t a = 0; a < cols; ++a) slope[k * cols + a] = (y[(k + 1) * cols + a] - y[k * cols + a]) / dx[k];
  }
  // Banded A (3 x n): A[0] superdiagonal, A[1] diagonal, A[2] subdiagonal.
  Vec upper(n, 0.0), diagonal(n, 0.0), lower(n, 0.0), b(n * cols);
  for (int64_t k = 1; k < n - 1; ++k) diagonal[k] = 2.0 * (dx[k - 1] + dx[k]);
  for (int64_t k = 2; k < n; ++k) upper[k] = dx[k - 2];
  for (int64_t k = 0; k + 2 < n; ++k) lower[k] = dx[k + 1];
  for (int64_t k = 1; k < n - 1; ++k) {
    for (int64_t a = 0; a < cols; ++a)
      b[k * cols + a] = 3.0 * (dx[k] * slope[(k - 1) * cols + a] + dx[k - 1] * slope[k * cols + a]);
  }
  // Natural ends: second derivative zero (bc value 0.0).
  diagonal[0] = 2.0 * dx[0];
  upper[1] = dx[0];
  diagonal[n - 1] = 2.0 * dx[n - 2];
  lower[n - 2] = dx[n - 2];
  for (int64_t a = 0; a < cols; ++a) {
    b[a] = -0.5 * 0.0 * (dx[0] * dx[0]) + 3.0 * (y[cols + a] - y[a]);
    b[(n - 1) * cols + a] =
        0.5 * 0.0 * (dx[n - 2] * dx[n - 2]) + 3.0 * (y[(n - 1) * cols + a] - y[(n - 2) * cols + a]);
  }
  Vec dl(lower.begin(), lower.end() - 1), du(upper.begin() + 1, upper.end());
  gtsv(dl, diagonal, du, b, cols);
  return b;
}

double sign(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : (v == 0.0 ? 0.0 : v)); }

double pchip_edge(double h0, double h1, double m0, double m1) {
  double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
  const bool mask = sign(d) != sign(m0);
  const bool mask2 = (sign(m0) != sign(m1)) && (std::fabs(d) > 3.0 * std::fabs(m0));
  if (mask) d = 0.0;
  else if (mask2) d = 3.0 * m0;
  return d;
}

// scipy PchipInterpolator derivatives at the knots.
Vec pchip_slopes(const Vec& x, const Vec& y, int64_t cols) {
  const int64_t n = static_cast<int64_t>(x.size());
  Vec h(n - 1), m((n - 1) * cols), dk(n * cols, 0.0);
  for (int64_t k = 0; k < n - 1; ++k) {
    h[k] = x[k + 1] - x[k];
    for (int64_t a = 0; a < cols; ++a) m[k * cols + a] = (y[(k + 1) * cols + a] - y[k * cols + a]) / h[k];
  }
  if (n == 2) {
    for (int64_t a = 0; a < cols; ++a) dk[a] = dk[cols + a] = m[a];
    return dk;
  }
  for (int64_t k = 1; k < n - 1; ++k) {
    const double w1 = 2.0 * h[k] + h[k - 1];
    const double w2 = h[k] + 2.0 * h[k - 1];
    for (int64_t a = 0; a < cols; ++a) {
      const double left = m[(k - 1) * cols + a], right = m[k * cols + a];
      const bool condition = sign(right) != sign(left) || right == 0.0 || left == 0.0;
      if (condition) {
        dk[k * cols + a] = 0.0;
      } else {
        const double whmean = (w1 / left + w2 / right) / (w1 + w2);
        dk[k * cols + a] = 1.0 / whmean;
      }
    }
  }
  for (int64_t a = 0; a < cols; ++a) {
    dk[a] = pchip_edge(h[0], h[1], m[a], m[cols + a]);
    dk[(n - 1) * cols + a] = pchip_edge(h[n - 2], h[n - 3], m[(n - 2) * cols + a], m[(n - 3) * cols + a]);
  }
  return dk;
}

// ---- scipy RotationSpline ----

using Mat3 = std::array<double, 9>;

Mat3 skew(const double* v) { return {0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0}; }

Mat3 matmul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i * 3 + j] = a[i * 3 + 0] * b[0 * 3 + j] + a[i * 3 + 1] * b[1 * 3 + j] + a[i * 3 + 2] * b[2 * 3 + j];
  return r;
}

void matvec(const Mat3& a, const double* v, double* out) {
  for (int i = 0; i < 3; ++i) out[i] = a[i * 3 + 0] * v[0] + a[i * 3 + 1] * v[1] + a[i * 3 + 2] * v[2];
}

double norm3(const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

void cross(const double* a, const double* b, double* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

Mat3 rate_to_rotvec_dot(const double* rotvec) {
  const double nm = norm3(rotvec);
  const double k = nm > 1e-4 ? (1.0 - 0.5 * nm / std::tan(0.5 * nm)) / (nm * nm)
                             : 1.0 / 12.0 + 1.0 / 720.0 * (nm * nm);
  const Mat3 s = skew(rotvec), ss = matmul(s, s);
  Mat3 r{};
  for (int i = 0; i < 9; ++i) r[i] = ((i % 4 == 0 ? 1.0 : 0.0) + 0.5 * s[i]) + k * ss[i];
  return r;
}

Mat3 rotvec_dot_to_rate(const double* rotvec) {
  const double nm = norm3(rotvec);
  double k1, k2;
  if (nm > 1e-4) {
    k1 = (1.0 - std::cos(nm)) / (nm * nm);
    k2 = (nm - std::sin(nm)) / std::pow(nm, 3.0);
  } else {
    k1 = 0.5 - (nm * nm) / 24.0;
    k2 = 1.0 / 6.0 - (nm * nm) / 120.0;
  }
  const Mat3 s = skew(rotvec), ss = matmul(s, s);
  Mat3 r{};
  for (int i = 0; i < 9; ++i) r[i] = ((i % 4 == 0 ? 1.0 : 0.0) - k1 * s[i]) + k2 * ss[i];
  return r;
}

void nonlinear_term(const double* rv, const double* rvd, double* out) {
  const double nm = norm3(rv);
  const double dp = rv[0] * rvd[0] + rv[1] * rvd[1] + rv[2] * rvd[2];
  double cp[3], ccp[3], dccp[3];
  cross(rv, rvd, cp);
  cross(rv, cp, ccp);
  cross(rvd, cp, dccp);
  double k1, k2, k3;
  if (nm > 1e-4) {
    const double sn = std::sin(nm), cs = std::cos(nm);
    k1 = (-nm * sn - 2.0 * (cs - 1.0)) / std::pow(nm, 4.0);
    k2 = (-2.0 * nm + 3.0 * sn - nm * cs) / std::pow(nm, 5.0);
    k3 = (nm - sn) / std::pow(nm, 3.0);
  } else {
    k1 = 1.0 / 12.0 - (nm * nm) / 180.0;
    k2 = -1.0 / 60.0 + (nm * nm) / 12604.0;
    k3 = 1.0 / 6.0 - (nm * nm) / 120.0;
  }
  for (int i = 0; i < 3; ++i) out[i] = dp * (k1 * cp[i] + k2 * ccp[i]) + k3 * dccp[i];
}

void gbsv(int64_t n, int64_t kl, int64_t ku, Vec& ab, Vec& b) {
  const int64_t ldab = 2 * kl + ku + 1, kv = ku + kl;
  auto at = [&](int64_t row, int64_t col) -> double& { return ab[row + ldab * col]; };
  std::vector<int64_t> pivots(n);
  int64_t ju = 0;
  for (int64_t j = 0; j < n; ++j) {
    const int64_t km = std::min(kl, n - 1 - j);
    int64_t jp = 0;
    double best = std::fabs(at(kv, j));
    for (int64_t i = 1; i <= km; ++i) {
      if (std::fabs(at(kv + i, j)) > best) { best = std::fabs(at(kv + i, j)); jp = i; }
    }
    pivots[j] = jp + j;
    if (at(kv + jp, j) == 0.0) throw std::invalid_argument("singular matrix");
    ju = std::max(ju, std::min(j + ku + jp, n - 1));
    if (jp != 0) {
      for (int64_t c = 0; c <= ju - j; ++c) std::swap(at(kv + jp - c, j + c), at(kv - c, j + c));
    }
    if (km > 0) {
      const double reciprocal = 1.0 / at(kv, j);
      for (int64_t i = 1; i <= km; ++i) at(kv + i, j) = reciprocal * at(kv + i, j);
      for (int64_t c = 1; c <= ju - j; ++c) {
        const double y = at(kv - c, j + c);
        if (y != 0.0) {
          const double temp = -1.0 * y;
          for (int64_t i = 1; i <= km; ++i) at(kv - c + i, j + c) = at(kv - c + i, j + c) + at(kv + i, j) * temp;
        }
      }
    }
  }
  // Forward: L, with the row interchanges.
  for (int64_t j = 0; j + 1 < n; ++j) {
    const int64_t lm = std::min(kl, n - 1 - j);
    if (pivots[j] != j) std::swap(b[pivots[j]], b[j]);
    if (b[j] != 0.0) {
      const double temp = -1.0 * b[j];
      for (int64_t i = 1; i <= lm; ++i) b[j + i] = b[j + i] + at(kv + i, j) * temp;
    }
  }
  // Backward: U, a band of width kl+ku above the diagonal (dtbsv).
  const int64_t k = kl + ku;
  for (int64_t j = n - 1; j >= 0; --j) {
    if (b[j] != 0.0) {
      b[j] = b[j] / at(k, j);
      const double temp = b[j];
      for (int64_t i = j - 1; i >= std::max<int64_t>(0, j - k); --i) b[i] = b[i] - temp * at(k + i - j, j);
    }
  }
}

struct RotationSpline {
  Vec times;
  std::vector<Quat> rotations;
  Vec coefficients;  // ppoly layout, 3 columns

  RotationSpline(const Vec& knots, const std::vector<Quat>& raw) : times(knots) {
    const int64_t n = static_cast<int64_t>(knots.size());
    rotations.reserve(n);
    for (const Quat& q : raw) rotations.push_back(sp_from_quat(q));
    const int64_t m = n - 1;
    Vec dt(m), rotvecs(m * 3), rates(m * 3), rotvecs_dot;
    for (int64_t k = 0; k < m; ++k) {
      dt[k] = times[k + 1] - times[k];
      sp_as_rotvec(sp_compose(sp_inv(rotations[k]), rotations[k + 1]), &rotvecs[k * 3]);
      for (int a = 0; a < 3; ++a) rates[k * 3 + a] = rotvecs[k * 3 + a] / dt[k];
    }
    if (n == 2) {
      rotvecs_dot = rates;
    } else {
      solve_rates(dt, rates, rotvecs, rotvecs_dot);
    }
    coefficients.assign(m * 4 * 3, 0.0);
    for (int64_t k = 0; k < m; ++k) {
      for (int a = 0; a < 3; ++a) {
        const double rv = rotvecs[k * 3 + a], ar = rates[k * 3 + a], rd = rotvecs_dot[k * 3 + a];
        coefficients[(k * 4 + 0) * 3 + a] = (-2.0 * rv + dt[k] * ar + dt[k] * rd) / std::pow(dt[k], 3.0);
        coefficients[(k * 4 + 1) * 3 + a] = (3.0 * rv - 2.0 * dt[k] * ar - dt[k] * rd) / (dt[k] * dt[k]);
        coefficients[(k * 4 + 2) * 3 + a] = ar;
        coefficients[(k * 4 + 3) * 3 + a] = 0.0;
      }
    }
  }

  // scipy RotationSpline._solve_for_angular_rates; ``rates`` comes back as the
  // knot angular rates, ``rotvecs_dot`` as the segment rotation-vector rates.
  static void solve_rates(const Vec& dt, Vec& rates, const Vec& rotvecs, Vec& rotvecs_dot) {
    const int64_t m = static_cast<int64_t>(dt.size());
    const double first[3] = {rates[0], rates[1], rates[2]};
    std::vector<Mat3> a(m), a_inv(m);
    for (int64_t k = 0; k < m; ++k) {
      a[k] = rate_to_rotvec_dot(&rotvecs[k * 3]);
      a_inv[k] = rotvec_dot_to_rate(&rotvecs[k * 3]);
    }
    // Block-tridiagonal system of m-1 unknown 3-vectors in (5, 5) band form.
    const int64_t size = 3 * (m - 1), kl = 5, ku = 5, ldab = 2 * kl + ku + 1;
    Vec band(ldab * size, 0.0);
    auto set = [&](int64_t i, int64_t j, double value) { band[(kl + ku + i - j) + ldab * j] = value; };
    for (int64_t k = 0; k < m - 2; ++k) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          set(3 * (k + 1) + r, 3 * k + c, 2.0 * a_inv[k + 1][r * 3 + c] / dt[k + 1]);
          set(3 * k + r, 3 * (k + 1) + c, 2.0 * a[k + 1][r * 3 + c] / dt[k + 1]);
        }
      }
    }
    for (int64_t k = 0; k < m - 1; ++k) {
      const double diagonal = 4.0 * (1.0 / dt[k] + 1.0 / dt[k + 1]);
      for (int r = 0; r < 3; ++r) set(3 * k + r, 3 * k + r, diagonal);
    }
    Vec b0(size);
    for (int64_t k = 0; k < m - 1; ++k) {
      for (int r = 0; r < 3; ++r)
        b0[3 * k + r] = 6.0 * (rotvecs[k * 3 + r] * std::pow(dt[k], -2.0) +
                               rotvecs[(k + 1) * 3 + r] * std::pow(dt[k + 1], -2.0));
    }
    double product[3];
    matvec(a_inv[0], first, product);
    for (int r = 0; r < 3; ++r) b0[r] -= 2.0 / dt[0] * product[r];
    matvec(a[m - 1], &rates[(m - 1) * 3], product);
    for (int r = 0; r < 3; ++r) b0[size - 3 + r] -= 2.0 / dt[m - 1] * product[r];

    rotvecs_dot.assign(m * 3, 0.0);
    for (int iteration = 0; iteration < 10; ++iteration) {
      for (int64_t k = 0; k < m; ++k) matvec(a[k], &rates[k * 3], &rotvecs_dot[k * 3]);
      Vec b(size);
      for (int64_t k = 0; k < m - 1; ++k) {
        double delta_beta[3];
        nonlinear_term(&rotvecs[k * 3], &rotvecs_dot[k * 3], delta_beta);
        for (int r = 0; r < 3; ++r) b[3 * k + r] = b0[3 * k + r] - delta_beta[r];
      }
      Vec work = band;
      gbsv(size, kl, ku, work, b);
      bool converged = true;
      for (int64_t i = 0; i < size; ++i) {
        const double delta = std::fabs(b[i] - rates[i]);
        rates[i] = b[i];
        if (!(delta < 1e-9 * (1.0 + std::fabs(b[i])))) converged = false;
      }
      if (converged) break;
    }
    for (int64_t k = 0; k < m; ++k) matvec(a[k], &rates[k * 3], &rotvecs_dot[k * 3]);
    // Knot rates: the fixed first rate, then the solved interior ones.
    Vec shifted(m * 3);
    for (int r = 0; r < 3; ++r) shifted[r] = first[r];
    for (int64_t i = 0; i < 3 * (m - 1); ++i) shifted[3 + i] = rates[i];
    rates = shifted;
  }

  Quat operator()(double t) const {
    const int64_t m = static_cast<int64_t>(times.size()) - 1;
    const int64_t k = ppoly_interval(times, t);
    const double s = t - times[k];
    double rotvec[3];
    for (int a = 0; a < 3; ++a) {
      const double coeff[4] = {coefficients[(k * 4 + 0) * 3 + a], coefficients[(k * 4 + 1) * 3 + a],
                               coefficients[(k * 4 + 2) * 3 + a], coefficients[(k * 4 + 3) * 3 + a]};
      rotvec[a] = ppoly_value(coeff, s);
    }
    int64_t index = static_cast<int64_t>(std::upper_bound(times.begin(), times.end(), t) - times.begin()) - 1;
    index = std::min(std::max<int64_t>(index, 0), m - 1);
    return sp_compose(rotations[index], sp_from_rotvec(rotvec));
  }
};

Quat slerp(const Quat& left, const Quat& right, double weight) {
  const Quat first = so3_normalize(left);
  Quat second = so3_normalize(right);
  double dot = first.x * second.x + first.y * second.y + first.z * second.z + first.w * second.w;
  if (dot < 0.0) second = {-second.x, -second.y, -second.z, -second.w};
  dot = std::min(std::max(std::fabs(dot), 0.0), 1.0);
  Quat out;
  if (dot > 0.9995) {
    out = {first.x + weight * (second.x - first.x), first.y + weight * (second.y - first.y),
           first.z + weight * (second.z - first.z), first.w + weight * (second.w - first.w)};
  } else {
    const double angle = std::acos(dot), denominator = std::sin(angle);
    const double a = std::sin((1.0 - weight) * angle) / denominator, b = std::sin(weight * angle) / denominator;
    out = {a * first.x + b * second.x, a * first.y + b * second.y, a * first.z + b * second.z,
           a * first.w + b * second.w};
  }
  return so3_normalize(out);
}

// ---- runtime resample_uniform ----

enum Kind { kLinear = 0, kHold = 1, kPose = 2 };
enum Method { kLinearMethod = 0, kPchip = 1, kCubic = 2 };

int64_t resampled_horizon(int64_t length, double source_hz, double target_hz) {
  return std::max<int64_t>(1, static_cast<int64_t>(std::floor(length * target_hz / source_hz + 0.5)));
}

int64_t output_width(int kind, int64_t width) { return kind == kPose ? 7 : width; }

// One row: float32 [T, D] in, float32 [count, D'] out.
void resample_row(const double* input, int64_t steps, int64_t width, double source_hz, double target_hz,
                  int kind, int64_t count, bool identity, int method, float* output) {
  Vec values(steps * width);
  for (int64_t i = 0; i < steps * width; ++i) values[i] = input[i];
  const int64_t out_width = output_width(kind, width);
  if (identity) {
    for (int64_t i = 0; i < steps * width; ++i) output[i] = static_cast<float>(values[i]);
    return;
  }
  Vec target_time(count), source_time(steps), interpolation_time(count);
  for (int64_t i = 0; i < count; ++i) target_time[i] = static_cast<double>(i) / target_hz;
  for (int64_t i = 0; i < steps; ++i) source_time[i] = static_cast<double>(i) / source_hz;
  std::vector<int64_t> left(count), right(count);
  Vec weight(count);
  for (int64_t i = 0; i < count; ++i) {
    interpolation_time[i] = std::min(target_time[i], source_time[steps - 1]);
    const double position = target_time[i] * source_hz;
    left[i] = std::min(static_cast<int64_t>(std::floor(position)), steps - 1);
    right[i] = std::min(left[i] + 1, steps - 1);
    weight[i] = position - static_cast<double>(left[i]);
  }
  Vec result(count * out_width);
  if (kind == kHold) {
    for (int64_t i = 0; i < count; ++i)
      for (int64_t a = 0; a < width; ++a) result[i * out_width + a] = values[left[i] * width + a];
  } else {
    const bool pose = kind == kPose;
    const int64_t cols = pose ? 3 : width;
    Vec continuous(steps * cols);
    for (int64_t t = 0; t < steps; ++t)
      for (int64_t a = 0; a < cols; ++a) continuous[t * cols + a] = values[t * width + a];
    if (steps == 1) {
      for (int64_t i = 0; i < count; ++i)
        for (int64_t a = 0; a < cols; ++a) result[i * out_width + a] = continuous[a];
    } else if (method == kPchip) {
      const Vec c = hermite_coefficients(source_time, continuous, pchip_slopes(source_time, continuous, cols), cols);
      ppoly_evaluate(source_time, c, cols, interpolation_time, out_width, 0, result.data());
    } else if (method == kCubic) {
      const Vec c = hermite_coefficients(source_time, continuous, natural_cubic_slopes(source_time, continuous, cols), cols);
      ppoly_evaluate(source_time, c, cols, interpolation_time, out_width, 0, result.data());
      // Guard: fall back to PCHIP on axes that materially leave their envelope.
      std::vector<int64_t> fallback;
      for (int64_t a = 0; a < cols; ++a) {
        double lower = continuous[a], upper = continuous[a];
        for (int64_t t = 1; t < steps; ++t) {
          lower = std::min(lower, continuous[t * cols + a]);
          upper = std::max(upper, continuous[t * cols + a]);
        }
        const double tolerance = 1e-9 + 0.05 * (upper - lower);
        for (int64_t i = 0; i < count; ++i) {
          const double v = result[i * out_width + a];
          if (v < lower - tolerance || v > upper + tolerance) { fallback.push_back(a); break; }
        }
      }
      if (!fallback.empty()) {
        const int64_t f = static_cast<int64_t>(fallback.size());
        Vec subset(steps * f);
        for (int64_t t = 0; t < steps; ++t)
          for (int64_t j = 0; j < f; ++j) subset[t * f + j] = continuous[t * cols + fallback[j]];
        const Vec c = hermite_coefficients(source_time, subset, pchip_slopes(source_time, subset, f), f);
        Vec sampled(count * f);
        ppoly_evaluate(source_time, c, f, interpolation_time, f, 0, sampled.data());
        for (int64_t i = 0; i < count; ++i)
          for (int64_t j = 0; j < f; ++j) result[i * out_width + fallback[j]] = sampled[i * f + j];
      }
    } else {
      for (int64_t i = 0; i < count; ++i)
        for (int64_t a = 0; a < cols; ++a) {
          const double lo = values[left[i] * width + a], hi = values[right[i] * width + a];
          result[i * out_width + a] = lo + weight[i] * (hi - lo);
        }
    }
    if (pose) {
      std::vector<Quat> quaternions(steps);
      for (int64_t t = 0; t < steps; ++t)
        quaternions[t] = {values[t * width + 3], values[t * width + 4], values[t * width + 5], values[t * width + 6]};
      auto store = [&](int64_t i, const Quat& q) {
        result[i * out_width + 3] = q.x;
        result[i * out_width + 4] = q.y;
        result[i * out_width + 5] = q.z;
        result[i * out_width + 6] = q.w;
      };
      if (steps == 1) {
        for (int64_t i = 0; i < count; ++i) store(i, quaternions[0]);
      } else if (method == kLinearMethod) {
        for (int64_t i = 0; i < count; ++i) store(i, slerp(quaternions[left[i]], quaternions[right[i]], weight[i]));
      } else {
        const RotationSpline spline(source_time, quaternions);
        for (int64_t i = 0; i < count; ++i) store(i, spline(interpolation_time[i]));
      }
    }
  }
  for (int64_t i = 0; i < count * out_width; ++i) output[i] = static_cast<float>(result[i]);
}

int parse_kind(const std::string& kind) {
  if (kind == "linear") return kLinear;
  if (kind == "hold") return kHold;
  if (kind == "pose_xyzw") return kPose;
  throw std::invalid_argument("unsupported resampling kind");
}

int parse_method(const std::string& method) {
  if (method == "linear") return kLinearMethod;
  if (method == "pchip") return kPchip;
  if (method == "cubic") return kCubic;
  throw std::invalid_argument("method must be one of ['cubic', 'linear', 'pchip']");
}

using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check_finite(const F64& values) {
  const double* data = values.data();
  for (py::ssize_t i = 0; i < values.size(); ++i)
    if (!std::isfinite(data[i])) throw std::invalid_argument("values must be a non-empty finite [T, D] array");
}

F32 resample_uniform_batch(const F64& values, double source_hz, double target_hz, const std::string& kind_name,
                           int64_t target_length, const std::string& method_name, int threads) {
  if (values.ndim() != 3 || values.shape(0) < 1 || values.shape(1) < 1)
    throw std::invalid_argument("values must have non-empty shape [B, T, D]");
  if (!(std::isfinite(source_hz) && source_hz > 0.0)) throw std::invalid_argument("source_hz must be positive and finite");
  if (!(std::isfinite(target_hz) && target_hz > 0.0)) throw std::invalid_argument("target_hz must be positive and finite");
  check_finite(values);
  const int kind = parse_kind(kind_name), method = parse_method(method_name);
  const int64_t batch = values.shape(0), steps = values.shape(1), width = values.shape(2);
  if (kind == kPose && width != 7) throw std::invalid_argument("pose_xyzw values must have shape [T, 7]");
  const int64_t standard = resampled_horizon(steps, source_hz, target_hz);
  const bool identity = source_hz == target_hz && (target_length < 0 || target_length == standard);
  const int64_t count = identity ? steps : (target_length < 0 ? standard : target_length);
  if (count < 1) throw std::invalid_argument("target_length must be a positive integer");
  const int64_t out_width = output_width(kind, width);
  F32 output({batch, count, out_width});
  const double* in = values.data();
  float* out = output.mutable_data();
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t row) {
      resample_row(in + row * steps * width, steps, width, source_hz, target_hz, kind, count, identity, method,
                   out + row * count * out_width);
    });
  }
  return output;
}

// Elementwise pose maps over [B, T, D] float32 arrays.
template <class Row>
F32 map_rows(const F64& values, int64_t in_width, int64_t out_width, int threads, Row&& row) {
  if (values.ndim() != 3 || values.shape(2) != in_width || values.shape(1) < 1)
    throw std::invalid_argument("expected shape [B, T, " + std::to_string(in_width) + "]");
  check_finite(values);
  const int64_t batch = values.shape(0), steps = values.shape(1);
  F32 output({batch, steps, out_width});
  const double* in = values.data();
  float* out = output.mutable_data();
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t b) {
      row(in + b * steps * in_width, steps, out + b * steps * out_width);
    });
  }
  return output;
}

// runtime integrate_eef_deltas: [T, 6] body increments -> [T, 7] absolute poses.
F32 integrate_eef_deltas_batch(const F64& deltas, int threads) {
  return map_rows(deltas, 6, 7, threads, [](const double* in, int64_t steps, float* out) {
    double position[3] = {0.0, 0.0, 0.0};
    Quat current{0.0, 0.0, 0.0, 1.0};
    for (int64_t t = 0; t < steps; ++t) {
      for (int a = 0; a < 3; ++a) {
        position[a] = position[a] + static_cast<double>(in[t * 6 + a]);
        out[t * 7 + a] = static_cast<float>(position[a]);
      }
      const Quat increment = so3_from_rotvec(in[t * 6 + 3], in[t * 6 + 4], in[t * 6 + 5]);
      current = so3_multiply(current, increment);
      out[t * 7 + 3] = static_cast<float>(current.x);
      out[t * 7 + 4] = static_cast<float>(current.y);
      out[t * 7 + 5] = static_cast<float>(current.z);
      out[t * 7 + 6] = static_cast<float>(current.w);
    }
  });
}

// runtime eef_pose_from_components(..., "rotation_vector").
F32 eef_pose_from_rotvec_batch(const F64& poses, int threads) {
  return map_rows(poses, 6, 7, threads, [](const double* in, int64_t steps, float* out) {
    for (int64_t t = 0; t < steps; ++t) {
      for (int a = 0; a < 3; ++a) out[t * 7 + a] = in[t * 6 + a];
      const Quat q = so3_from_rotvec(in[t * 6 + 3], in[t * 6 + 4], in[t * 6 + 5]);
      out[t * 7 + 3] = static_cast<float>(q.x);
      out[t * 7 + 4] = static_cast<float>(q.y);
      out[t * 7 + 5] = static_cast<float>(q.z);
      out[t * 7 + 6] = static_cast<float>(q.w);
    }
  });
}

// runtime eef_pose_absolute: [T, 7] xyz + xyzw -> [T, 6] xyz + rotation vector.
F32 eef_pose_absolute_batch(const F64& poses, int threads) {
  return map_rows(poses, 7, 6, threads, [](const double* in, int64_t steps, float* out) {
    for (int64_t t = 0; t < steps; ++t) {
      for (int a = 0; a < 3; ++a) out[t * 6 + a] = in[t * 7 + a];
      double rotvec[3];
      so3_to_rotvec({in[t * 7 + 3], in[t * 7 + 4], in[t * 7 + 5], in[t * 7 + 6]}, rotvec);
      for (int a = 0; a < 3; ++a) out[t * 6 + 3 + a] = static_cast<float>(rotvec[a]);
    }
  });
}

// runtime eef_pose_deltas: local [T, 7] path -> [T, 6] body increments.
F32 eef_pose_deltas_batch(const F64& poses, int threads) {
  return map_rows(poses, 7, 6, threads, [](const double* in, int64_t steps, float* out) {
    Quat previous{0.0, 0.0, 0.0, 1.0};
    for (int64_t t = 0; t < steps; ++t) {
      for (int a = 0; a < 3; ++a) {
        const double before = t == 0 ? 0.0 : static_cast<double>(in[(t - 1) * 7 + a]);
        out[t * 6 + a] = static_cast<float>(static_cast<double>(in[t * 7 + a]) - before);
      }
      const Quat q = so3_normalize({in[t * 7 + 3], in[t * 7 + 4], in[t * 7 + 5], in[t * 7 + 6]});
      double rotvec[3];
      so3_to_rotvec(so3_multiply({-previous.x, -previous.y, -previous.z, previous.w}, q), rotvec);
      for (int a = 0; a < 3; ++a) out[t * 6 + 3 + a] = static_cast<float>(rotvec[a]);
      previous = q;
    }
  });
}

// Re-express per-step rotation increments between world-frame (left) and
// body-frame (right) multiplication, as scipy ``Rotation`` composes them.
F32 rotvec_left_to_right_batch(const F64& increments, int threads) {
  return map_rows(increments, 3, 3, threads, [](const double* in, int64_t steps, float* out) {
    Quat current{0.0, 0.0, 0.0, 1.0};
    for (int64_t t = 0; t < steps; ++t) {
      const double v[3] = {in[t * 3], in[t * 3 + 1], in[t * 3 + 2]};
      const Quat following = sp_compose(sp_from_rotvec(v), current);
      double rotvec[3];
      sp_as_rotvec(sp_compose(sp_inv(current), following), rotvec);
      for (int a = 0; a < 3; ++a) out[t * 3 + a] = static_cast<float>(rotvec[a]);
      current = following;
    }
  });
}

F32 rotvec_right_to_left_batch(const F64& increments, int threads) {
  return map_rows(increments, 3, 3, threads, [](const double* in, int64_t steps, float* out) {
    Quat current{0.0, 0.0, 0.0, 1.0};
    for (int64_t t = 0; t < steps; ++t) {
      const double v[3] = {in[t * 3], in[t * 3 + 1], in[t * 3 + 2]};
      const Quat following = sp_compose(current, sp_from_rotvec(v));
      double rotvec[3];
      sp_as_rotvec(sp_compose(following, sp_inv(current)), rotvec);
      for (int a = 0; a < 3; ++a) out[t * 3 + a] = static_cast<float>(rotvec[a]);
      current = following;
    }
  });
}

}  // namespace

void register_resample(py::module_& module) {
  module.def("resample_uniform_batch", &resample_uniform_batch, py::arg("values"), py::arg("source_hz"),
             py::arg("target_hz"), py::arg("kind"), py::arg("target_length") = -1,
             py::arg("method") = "linear", py::arg("threads") = 0,
             "runtime resample_uniform over each [T, D] row of a [B, T, D] batch.");
  module.def("integrate_eef_deltas_batch", &integrate_eef_deltas_batch, py::arg("deltas"), py::arg("threads") = 0);
  module.def("eef_pose_from_rotvec_batch", &eef_pose_from_rotvec_batch, py::arg("poses"), py::arg("threads") = 0);
  module.def("eef_pose_absolute_batch", &eef_pose_absolute_batch, py::arg("poses"), py::arg("threads") = 0);
  module.def("eef_pose_deltas_batch", &eef_pose_deltas_batch, py::arg("poses"), py::arg("threads") = 0);
  module.def("rotvec_left_to_right_batch", &rotvec_left_to_right_batch, py::arg("increments"),
             py::arg("threads") = 0);
  module.def("rotvec_right_to_left_batch", &rotvec_right_to_left_batch, py::arg("increments"),
             py::arg("threads") = 0);
}
