import numpy as np
import cmath

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Transform:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def apply(self, p):
        # Standard 2D transformation (before QCQP)
        new_x = p.x * np.cos(self.theta) - p.y * np.sin(self.theta) + self.x
        new_y = p.x * np.sin(self.theta) + p.y * np.cos(self.theta) + self.y
        return Point(new_x, new_y)

def transform_points(points, t):
    return [t.apply(p) for p in points]

def get_cubic_root(a, b, c, d):
    """
    Conceptually, to enforce the unit-circle constraint sin^2(theta) + cos^2(theta) = 1
    In QCQP, enforcing the cnstraint introduces a Lagrange multiplier
    """
    # Reduce to depressed cubic
    p = c/a - (b**2)/(3*a**2)
    q = (2*b**3)/(27*a**3) + d/a - (b*c)/(3*a**2)

    xi = complex(-0.5, np.sqrt(3)/2)
    inside = cmath.sqrt(q**2/4 + p**3/27)

    root = complex(0, 0)
    for k in range(3):
        # Cardano's Formula for 3 possible roots
        term1 = (xi**k) * ((-q/2.0 + inside)**(1.0/3.0))
        term2 = (xi**(2.0*k)) * ((-q/2.0 - inside)**(1.0/3.0))
        root = -b/(3*a) + term1 + term2
        
        if abs(root.imag) > 1e-6: # Returning first complex root as per C++ logic
            return root
    return root

def greatest_real_root(a, b, c, d, e):
    """
    Same as above... don't *really* need to understand
    """
    # Depressed Quartic reduction
    p = (8*a*c - 3*b**2) / (8*a**2)
    q = (b**3 - 4*a*b*c + 8*a**2*d) / (8*a**3)
    r = (-3*b**4 + 256*a**3*e - 64*a**2*b*d + 16*a*b**2*c) / (256*a**4)

    # Ferrari's Solution: 8m^3 + 8pm^2 + (2p^2-8r)m - q^2 = 0
    m_complex = get_cubic_root(8.0, 8.0*p, 2.0*p**2 - 8.0*r, -q**2)
    m = m_complex.real # Following the logic of finding the real component

    # Calculating the 4 roots
    sqrt_2m = cmath.sqrt(2.0 * m)
    part1 = -(2*p + 2.0*m)
    part2 = (np.sqrt(2.0) * q) / sqrt_2m

    roots = [
        -b/(4*a) + (sqrt_2m + cmath.sqrt(part1 - part2)) / 2.0,
        -b/(4*a) + (sqrt_2m - cmath.sqrt(part1 - part2)) / 2.0,
        -b/(4*a) + (-sqrt_2m + cmath.sqrt(part1 + part2)) / 2.0,
        -b/(4*a) + (-sqrt_2m - cmath.sqrt(part1 + part2)) / 2.0
    ]

    max_real_root = 0.0
    for root in roots:
        if abs(root.imag) < 1e-6:
            max_real_root = max(max_real_root, root.real)
            
    return max_real_root

def update_transform(correspondences, curr_trans):
    """
    QCQP-based Point-to-Line ICP (Global Pose Calibration)
    Solves:
        min_x  x^T M x - 2 g^T x
        s.t.   x^T W x = 1
    where x = [tx, ty, cosθ, sinθ]
    """

    number_iter = 1

    for _ in range(number_iter):

        # Quadratic objective components
        M = np.zeros((4, 4))
        g = np.zeros((4, 1))

        # Rotation constraint matrix
        W = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # -----------------------------
        # Build M and g from data
        # -----------------------------
        for corr in correspondences:

            p = corr.p
            pi = corr.pi
            n = corr.n.reshape(2, 1)

            # M_i matrix (maps x -> transformed point)
            Mi = np.array([
                [1.0, 0.0,  p.x, -p.y],
                [0.0, 1.0,  p.y,  p.x]
            ])

            # C_i = n n^T (normal projection)
            Ci = n @ n.T

            # π_i vector
            pi_vec = np.array([[pi.x], [pi.y]])

            # Accumulate quadratic terms
            M += Mi.T @ Ci @ Mi
            g += Mi.T @ Ci @ pi_vec

        # --------------------------------
        # Block partition M
        # --------------------------------
        A = M[0:2, 0:2]
        B = M[0:2, 2:4]
        D = M[2:4, 2:4]
        g1 = g[0:2]
        g2 = g[2:4]

        # --------------------------------
        # Eliminate translation analytically
        # --------------------------------
        A_inv = np.linalg.inv(A)

        S = D - B.T @ A_inv @ B
        h = g2 - B.T @ A_inv @ g1

        # --------------------------------
        # Solve QCQP via Lagrange multiplier
        # --------------------------------
        # We solve:
        #   (S - λ I) r = h
        #   r^T r = 1
        #
        # Leads to a quartic polynomial in λ

        # Coefficients of det(S - λI)
        a = 1.0
        b = -np.trace(S)
        c = np.linalg.det(S) + np.trace(S) * 0  # simplified for 2x2
        d = -np.linalg.det(S) * 0
        e = -np.linalg.norm(h)**2

        lambda_val = greatest_real_root(a, b, c, d, e)

        # --------------------------------
        # Recover rotation variables
        # --------------------------------
        r = np.linalg.solve(S - lambda_val * np.eye(2), h)

        # Normalize to enforce unit circle
        r = r / np.linalg.norm(r)

        # --------------------------------
        # Recover translation
        # --------------------------------
        t = A_inv @ (g1 - B @ r)

        # --------------------------------
        # Assemble full solution
        # --------------------------------
        x = np.zeros(4)
        x[0] = t[0, 0]        # tx
        x[1] = t[1, 0]        # ty
        x[2] = r[0, 0]        # cosθ
        x[3] = r[1, 0]        # sinθ

        # Convert back to angle
        curr_trans.x = x[0]
        curr_trans.y = x[1]
        curr_trans.theta = np.arctan2(x[3], x[2])

    return curr_trans
