struct Float4 {
		union {
			Float v[4];
			__m128 m;
		};

		explicit Float4(Float x) { for (int32_t i = 0; i < 4; i++)v[i] = x; }

		Float4() = default;

		explicit Float4(__m128 x) { m = x; }

		inline Float4& operator+=(const Float4& rhs) {
			v[0] += rhs.v[0];
			v[1] += rhs.v[1];
			v[2] += rhs.v[2];
			v[3] += rhs.v[3];
			return *this;
		}

		inline Float4& operator-=(const Float4& rhs) {
			v[0] -= rhs.v[0];
			v[1] -= rhs.v[1];
			v[2] -= rhs.v[2];
			v[3] -= rhs.v[3];
			return *this;
		}

		inline Float4& operator*=(const Float4& rhs) {
			v[0] *= rhs.v[0];
			v[1] *= rhs.v[1];
			v[2] *= rhs.v[2];
			v[3] *= rhs.v[3];
			return *this;
		}

		inline Float4& operator/=(const Float4& rhs) {
			v[0] /= rhs.v[0];
			v[1] /= rhs.v[1];
			v[2] /= rhs.v[2];
			v[3] /= rhs.v[3];
			return *this;
		}

		inline Float4 operator+(const Float4& rhs) const {
			return Float4(_mm_add_ps(m, rhs.m));

		}

		Float4 operator-(const Float4& rhs) const {
			return Float4(_mm_sub_ps(m, rhs.m));

		}

		Float4 operator*(const Float4& rhs) const {
			return Float4(_mm_mul_ps(m, rhs.m));

		}

		Float4 operator/(const Float4& rhs) const {
			return Float4(_mm_div_ps(m, rhs.m));

		}

		Float4 operator<(const Float4& rhs) {
			return Float4(_mm_cmplt_ps(m, rhs.m));
		}

		Float4 operator<=(const Float4& rhs) {
			return Float4(_mm_cmple_ps(m, rhs.m));
		}
		Float4 operator==(const Float4& rhs) {
			return Float4(_mm_cmpeq_ps(m, rhs.m));
		}
		Float4 operator!=(const Float4& rhs) {
			return Float4(_mm_cmpneq_ps(m, rhs.m));
		}
		Float4 operator>(const Float4& rhs) {
			return Float4(_mm_cmpgt_ps(m, rhs.m));
		}

		Float4 operator>=(const Float4& rhs) {
			return Float4(_mm_cmpge_ps(m, rhs.m));
		}

		Float4 operator&&(const Float4& rhs) {
			return Float4(_mm_and_ps(m, rhs.m));
		}

		void inv() {
			v[0] = 1 / v[0];
			v[1] = 1 / v[1];
			v[2] = 1 / v[2];
			v[3] = 1 / v[3];
		}

		void abs() {
			v[0] = fabsf(v[0]);
			v[1] = fabsf(v[1]);
			v[2] = fabsf(v[2]);
			v[3] = fabsf(v[3]);
		}

		Float& operator[](size_t i) { return v[i]; }

		const Float& operator[](size_t i) const { return v[i]; }

		static constexpr size_t width() { return 4; }
	};
 

template<class T, size_t N>
    struct VecBase {
        static constexpr size_t _N = N;
        T _v[_N];

        VecBase() {
            for (auto &i:_v) {
                i = T();
            }
        }
    };

    template<class T>
    struct VecBase<T, 2> {
        static constexpr size_t _N = 2;
        union {
            T _v[_N];
            struct {
                T x, y;
            };
        };

        VecBase() : x(T()), y(T()) {}

        VecBase(const T &x, const T &y) : x(x), y(y) {}
    };

    template<class T>
    struct VecBase<T, 3> {
        static constexpr size_t _N = 3;
        union {
            T _v[_N];
            struct {
                T x, y, z;
            };
        };

        VecBase() : x(T()), y(T()), z(T()) {}

        VecBase(const T &x, const T &y, const T &z) : x(x), y(y), z(z) {}
    };

    template<class T>
    struct VecBase<T, 4> {
        static constexpr size_t _N = 4;
        union {
            T _v[_N];
            struct {
                T x, y, z, w;
            };
        };

        VecBase() : x(T()), y(T()), z(T()), w(T()) {}

        VecBase(const T &x, const T &y, const T &z, const T &w) : x(x), y(y), z(z), w(w) {}
    };

    template<class T, size_t N>
    struct Vec : VecBase<T, N> {
        using VecBase<T, N>::VecBase;
        static constexpr size_t _N = VecBase<T, N>::_N;

        Vec &operator+=(const Vec &rhs) {
            for (size_t i = 0; i < _N; i++) {
                this->_v[i] += rhs._v[i];
            }
            return *this;
        }

        Vec &operator-=(const Vec &rhs) {
            for (size_t i = 0; i < _N; i++) {
                this->_v[i] -= rhs._v[i];
            }
            return *this;
        }

        Vec &operator/=(const Vec &rhs) {
            for (size_t i = 0; i < _N; i++) {
                this->_v[i] /= rhs._v[i];
            }
            return *this;
        }

        Vec &operator*=(const Vec &rhs) {
            for (size_t i = 0; i < _N; i++) {
                this->_v[i] *= rhs._v[i];
            }
            return *this;
        }

        Vec &operator/=(const T &rhs) {
            for (size_t i = 0; i < _N; i++) {
                this->_v[i] /= rhs;
            }
            return *this;
        }

        Vec &operator*=(const T &rhs) {
            for (size_t i = 0; i < _N; i++) {
                this->_v[i] *= rhs;
            }
            return *this;
        }

        Vec operator+(const Vec &rhs) const {
            Vec tmp = *this;
            tmp += rhs;
            return tmp;
        }

        Vec operator-(const Vec &rhs) const {
            Vec tmp = *this;
            tmp -= rhs;
            return tmp;
        }

        Vec operator*(const Vec &rhs) const {
            Vec tmp = *this;
            tmp *= rhs;
            return tmp;
        }

        Vec operator/(const Vec &rhs) const {
            Vec tmp = *this;
            tmp /= rhs;
            return tmp;
        }

        Vec operator*(const T &rhs) const {
            Vec tmp = *this;
            tmp *= rhs;
            return tmp;
        }

        Vec operator/(const T &rhs) const {
            Vec tmp = *this;
            tmp /= rhs;
            return tmp;
        }

        T dot(const Vec &rhs) const {
            T sum = this->_v[0] * rhs._v[0];
            for (size_t i = 1; i < N; i++) {
                sum += this->_v[i] * rhs._v[i];
            }
            return sum;
        }

        T absDot(const Vec &rhs) const {
            return std::abs(dot(rhs));
        }

        T lengthSquared() const {
            return dot(*this);
        }

        T length() const {
            return std::sqrt(lengthSquared());
        }

        void normalize() {
            (*this) /= length();
        }

        Vec<T, N> normalized() const {
            auto t = *this;
            t.normalize();
            return t;
        }

        Vec<T, N> operator-() const {
            auto tmp = *this;
            for (auto i = 0; i < N; i++) {
                tmp._v[i] = -tmp._v[i];
            }
            return tmp;
        }

        const T &operator[](size_t i) const {
            return this->_v[i];
        }

        T &operator[](size_t i) {
            return this->_v[i];
        }

        T max() const {
            T v = this->_v[0];
            for (int i = 1; i < N; i++) {
                v = std::max(v, this->_v[i]);
            }
            return v;
        }

        T min() const {
            T v = this->_v[0];
            for (int i = 1; i < N; i++) {
                v = std::min(v, this->_v[i]);
            }
            return v;
        }

        friend Vec<T, N> operator*(T k, const Vec<T, N> &rhs) {
            auto t = rhs;
            t *= k;
            return t;
        }
    };


    template<class T, size_t N>
    Vec<T, N> min(const Vec<T, N> &a, const Vec<T, N> &b) {
        Vec<T, N> r;
        for (int i = 0; i < N; i++) {
            r[i] = std::min(a[i], b[i]);
        }
        return r;
    }

    template<class T, size_t N>
    Vec<T, N> max(const Vec<T, N> &a, const Vec<T, N> &b) {
        Vec<T, N> r;
        for (int i = 0; i < N; i++) {
            r[i] = std::max(a[i], b[i]);
        }
        return r;
    }

    using Point3i =Vec<int, 3>;
    using Point3f = Vec<float, 3>;
    using Point2i = Vec<int, 2>;
    using Point2f = Vec<int, 2>;


    template<class T, size_t N>
    struct BoundBox {
        Vec<T, N> pMin, pMax;

        BoundBox unionOf(const BoundBox &box) const {
            return BoundBox{min(pMin, box.pMin), max(pMax, box.pMax)};
        }

        BoundBox unionOf(const Vec<T, N> &rhs) const {
            return BoundBox{min(pMin, rhs), max(pMax, rhs)};
        }

        Vec<T, N> centroid() const {
            return (pMin + pMax) * 0.5f;
        }

        Vec<T, N> size() const {
            return pMax - pMin;
        }

        T surfaceArea() const {
            auto a = (size()[0] * size()[1] + size()[0] * size()[2] + size()[1] * size()[2]);
            return a + a;
        }

        bool intersects(const BoundBox &rhs) const {
            for (size_t i = 0; i < N; i++) {
                if (pMin[i] > rhs.pMax[i] || pMax[i] < rhs.pMin[i]);
                else {
                    return true;
                }
            }
            return false;
        }

        Vec<T, N> offset(const Vec<T, N> &p) const {
            auto o = p - pMin;
            return o / size();
        }
    };

    using Bounds3f = BoundBox<float, 3>;

    struct Vec3f : Vec<float, 3> {

        Vec3f(const Vec<float, 3> &v) : Vec(v.x, v.y, v.z) {}

        Vec3f(float x, float y, float z) : Vec(x, y, z) {}

        Vec3f(float x = 0) : Vec(x, x, x) {}

        Vec3f cross(const Vec3f &v) const {
            return Vec3f(
                    y * v.z - z * v.y,
                    z * v.x - x * v.z,
                    x * v.y - y * v.x
            );
        }
    };

    struct Normal3f : public Vec3f {
        using Vec3f::Vec3f;
    };

    inline void ComputeLocalFrame(const Vec3f &v1, Vec3f *v2, Vec3f *v3) {
        if (std::abs(v1.x) > std::abs(v1.y))
            *v2 = Vec3f(-v1.z, 0, v1.x) /
                  std::sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            *v2 = Vec3f(0, v1.z, -v1.y) /
                  std::sqrt(v1.y * v1.y + v1.z * v1.z);
        *v3 = v1.cross(*v2).normalized();
    }

    struct CoordinateSystem {
        CoordinateSystem() = default;

        explicit CoordinateSystem(const Vec3f &v) : normal(v) {
            ComputeLocalFrame(v, &localX, &localZ);
        }

        Vec3f worldToLocal(const Vec3f &v) const {
            return Vec3f(localX.dot(v), normal.dot(v), localZ.dot(v));
        }

        Vec3f localToWorld(const Vec3f &v) const {
            return Vec3f(v.x * localX + v.y * normal + v.z * localZ);
        }

    private:
        Vec3f normal;
        Vec3f localX, localZ;
    };

    struct Matrix4 {
        Matrix4() = default;

        static Matrix4 identity() {
            float i[4][4] = {
                    {1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0},
                    {0, 0, 0, 1},
            };
            return Matrix4(i);
        }

        static Matrix4 translate(const Vec3f &v) {
            float t[4][4] = {
                    {1, 0, 0, v.x},
                    {0, 1, 0, v.y},
                    {0, 0, 1, v.z},
                    {0, 0, 0, 1},
            };
            return Matrix4(t);
        }

        static Matrix4 scale(const Vec3f &v) {
            float s[4][4] = {
                    {v.x, 0,   0,   0},
                    {0,   v.y, 0,   0},
                    {0,   0,   v.z, 0},
                    {0,   0,   0,   1},
            };
            return Matrix4(s);
        }

        static Matrix4 rotate(const Vec3f &x, const Vec3f &axis, const Float angle) {
            const Float s = sin(angle);
            const Float c = cos(angle);
            const Float oc = Float(1.0) - c;
            float r[4][4] = {
                    {oc * axis.x * axis.x + c,
                            oc * axis.x * axis.y - axis.z * s,
                               oc * axis.z * axis.x + axis.y * s, 0},
                    {oc * axis.x * axis.y + axis.z * s,
                            oc * axis.y * axis.y + c,
                               oc * axis.y * axis.z - axis.x * s, 0},
                    {oc * axis.z * axis.x - axis.y * s,
                            oc * axis.y * axis.z + axis.x * s,
                               oc * axis.z * axis.z + c,          0},
                    {0,     0, 0,                                 1}
            };
            return Matrix4(r);
        }

        static Matrix4 lookAt(const Vec3f &from, const Vec3f &to) {
            Vec3f up(0, 1, 0);
            Vec3f d = to - from;
            d.normalize();
            Vec3f xAxis = up.cross(d).normalized();
            Vec3f yAxis = xAxis.cross(d).normalized();
            float m[4][4] = {
                    {xAxis.x, yAxis.x, d.x, 0},
                    {xAxis.y, yAxis.y, d.y, 0},
                    {xAxis.z, yAxis.z, d.z, 0},
                    {0,       0,       0,   1}
            };
            return Matrix4::translate(from) * Matrix4(m);
        }

        explicit Matrix4(float data[4][4]) {
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    _rows[i][j] = data[i][j];
                }
            }
        }

        Matrix4 operator*(const Matrix4 &rhs) const {
            Matrix4 m = *this;
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    m._rows[i][j] = _rows[i].dot(rhs.column(j));
                }
            }
            return m;
        }

        Matrix4 &operator*=(const Matrix4 &rhs) {
            auto m = *this * rhs;
            for (int i = 0; i < 4; i++)
                _rows[i] = m._rows[i];
            return *this;
        }

        Vec<float, 4> operator*(const Vec<float, 4> &v) const {
            return Vec<float, 4>{
                    _rows[0].dot(v),
                    _rows[1].dot(v),
                    _rows[2].dot(v),
                    _rows[3].dot(v)};
        }

        Vec<float, 4> column(size_t i) const {
            return Vec<float, 4>{_rows[0][i], _rows[1][i], _rows[2][i], _rows[3][i]};
        }

        Matrix4 inverse(bool *suc = nullptr) const {
            auto m = reinterpret_cast<const float *>(_rows);
            float inv[16], det;
            int i;

            inv[0] = m[5] * m[10] * m[15] -
                     m[5] * m[11] * m[14] -
                     m[9] * m[6] * m[15] +
                     m[9] * m[7] * m[14] +
                     m[13] * m[6] * m[11] -
                     m[13] * m[7] * m[10];

            inv[4] = -m[4] * m[10] * m[15] +
                     m[4] * m[11] * m[14] +
                     m[8] * m[6] * m[15] -
                     m[8] * m[7] * m[14] -
                     m[12] * m[6] * m[11] +
                     m[12] * m[7] * m[10];

            inv[8] = m[4] * m[9] * m[15] -
                     m[4] * m[11] * m[13] -
                     m[8] * m[5] * m[15] +
                     m[8] * m[7] * m[13] +
                     m[12] * m[5] * m[11] -
                     m[12] * m[7] * m[9];

            inv[12] = -m[4] * m[9] * m[14] +
                      m[4] * m[10] * m[13] +
                      m[8] * m[5] * m[14] -
                      m[8] * m[6] * m[13] -
                      m[12] * m[5] * m[10] +
                      m[12] * m[6] * m[9];

            inv[1] = -m[1] * m[10] * m[15] +
                     m[1] * m[11] * m[14] +
                     m[9] * m[2] * m[15] -
                     m[9] * m[3] * m[14] -
                     m[13] * m[2] * m[11] +
                     m[13] * m[3] * m[10];

            inv[5] = m[0] * m[10] * m[15] -
                     m[0] * m[11] * m[14] -
                     m[8] * m[2] * m[15] +
                     m[8] * m[3] * m[14] +
                     m[12] * m[2] * m[11] -
                     m[12] * m[3] * m[10];

            inv[9] = -m[0] * m[9] * m[15] +
                     m[0] * m[11] * m[13] +
                     m[8] * m[1] * m[15] -
                     m[8] * m[3] * m[13] -
                     m[12] * m[1] * m[11] +
                     m[12] * m[3] * m[9];

            inv[13] = m[0] * m[9] * m[14] -
                      m[0] * m[10] * m[13] -
                      m[8] * m[1] * m[14] +
                      m[8] * m[2] * m[13] +
                      m[12] * m[1] * m[10] -
                      m[12] * m[2] * m[9];

            inv[2] = m[1] * m[6] * m[15] -
                     m[1] * m[7] * m[14] -
                     m[5] * m[2] * m[15] +
                     m[5] * m[3] * m[14] +
                     m[13] * m[2] * m[7] -
                     m[13] * m[3] * m[6];

            inv[6] = -m[0] * m[6] * m[15] +
                     m[0] * m[7] * m[14] +
                     m[4] * m[2] * m[15] -
                     m[4] * m[3] * m[14] -
                     m[12] * m[2] * m[7] +
                     m[12] * m[3] * m[6];

            inv[10] = m[0] * m[5] * m[15] -
                      m[0] * m[7] * m[13] -
                      m[4] * m[1] * m[15] +
                      m[4] * m[3] * m[13] +
                      m[12] * m[1] * m[7] -
                      m[12] * m[3] * m[5];

            inv[14] = -m[0] * m[5] * m[14] +
                      m[0] * m[6] * m[13] +
                      m[4] * m[1] * m[14] -
                      m[4] * m[2] * m[13] -
                      m[12] * m[1] * m[6] +
                      m[12] * m[2] * m[5];

            inv[3] = -m[1] * m[6] * m[11] +
                     m[1] * m[7] * m[10] +
                     m[5] * m[2] * m[11] -
                     m[5] * m[3] * m[10] -
                     m[9] * m[2] * m[7] +
                     m[9] * m[3] * m[6];

            inv[7] = m[0] * m[6] * m[11] -
                     m[0] * m[7] * m[10] -
                     m[4] * m[2] * m[11] +
                     m[4] * m[3] * m[10] +
                     m[8] * m[2] * m[7] -
                     m[8] * m[3] * m[6];

            inv[11] = -m[0] * m[5] * m[11] +
                      m[0] * m[7] * m[9] +
                      m[4] * m[1] * m[11] -
                      m[4] * m[3] * m[9] -
                      m[8] * m[1] * m[7] +
                      m[8] * m[3] * m[5];

            inv[15] = m[0] * m[5] * m[10] -
                      m[0] * m[6] * m[9] -
                      m[4] * m[1] * m[10] +
                      m[4] * m[2] * m[9] +
                      m[8] * m[1] * m[6] -
                      m[8] * m[2] * m[5];

            det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

            if (det == 0) {
                if (suc) {
                    *suc = false;
                }
            }

            det = 1.0 / det;

            Matrix4 out;
            auto invOut = reinterpret_cast<float *>(out._rows);
            for (i = 0; i < 16; i++)
                invOut[i] = inv[i] * det;
            if (suc) {
                *suc = true;
            }
            return out;
        }

        Vec<float, 4> &operator[](int i) {
            return _rows[i];
        }

        const Vec<float, 4> &operator[](int i) const {
            return _rows[i];
        }

    private:
        Vec<float, 4> _rows[4];
        static_assert(sizeof(_rows) == sizeof(float) * 16, "Matrix4 must have packed 16 floats");
    };


    class Transform {
        Matrix4 m, invM;
    public:
        [[nodiscard]] const Matrix4 &matrix() const { return m; }

        explicit Transform(const Matrix4 &m = Matrix4::identity()) : m(m), invM(m.inverse()) {}

        explicit Transform(const Matrix4 &m, const Matrix4 &invM) : m(m), invM(invM) {}

        Transform operator*(const Transform &transform) {
            return Transform(m * transform.m);
        }

        [[nodiscard]] Transform inverse() const {
            return Transform(invM, m);
        }

        Vec3f operator()(const Vec3f &v) const {
            return Vec3f(
                    m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                    m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                    m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
        }

        Normal3f operator()(const Normal3f &v) const {
            return Normal3f(
                    invM[0][0] * v.x + invM[1][0] * v.y + invM[2][0] * v.z,
                    invM[0][1] * v.x + invM[1][1] * v.y + invM[2][1] * v.z,
                    invM[0][2] * v.x + invM[1][2] * v.y + invM[2][2] * v.z);
        }

        Point3f operator()(const Point3f &v) const {
            auto p = Point3f(
                    m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3],
                    m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3],
                    m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3]);
            auto w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
            if (w != 1) {
                p /= w;
            }
            return p;
        }
    };
