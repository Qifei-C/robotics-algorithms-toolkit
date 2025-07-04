import numpy as np


class Quaternion:
    """
    Quaternion class for 3D rotations
    
    Quaternion format: q = [w, x, y, z] where w is the scalar part
    """
    
    def __init__(self, w=1.0, vec=None):
        if vec is None:
            vec = [0.0, 0.0, 0.0]
        
        if isinstance(w, (list, np.ndarray)) and len(w) == 4:
            # Initialize from 4-element array
            self.q = np.array(w, dtype=float)
        else:
            # Initialize from scalar and vector
            self.q = np.array([w, vec[0], vec[1], vec[2]], dtype=float)
    
    @property
    def scalar(self):
        return self.q[0]
    
    @property
    def vec(self):
        return self.q[1:4]
    
    def normalize(self):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm
        return self
    
    def inv(self):
        """Compute quaternion inverse (conjugate for unit quaternions)"""
        q_inv = Quaternion()
        q_inv.q[0] = self.q[0]
        q_inv.q[1:4] = -self.q[1:4]
        return q_inv
    
    def __mul__(self, other):
        """Quaternion multiplication"""
        if isinstance(other, Quaternion):
            # Quaternion multiplication formula
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            return Quaternion([w, x, y, z])
        else:
            raise TypeError("Multiplication only defined between quaternions")
    
    def axis_angle(self):
        """Convert quaternion to axis-angle representation"""
        # Handle near-identity quaternions
        if abs(self.q[0]) >= 1.0:
            return np.zeros(3)
        
        # Compute angle
        angle = 2.0 * np.arccos(np.clip(self.q[0], -1.0, 1.0))
        
        # Compute axis
        s = np.sqrt(1.0 - self.q[0]**2)
        if s < 1e-10:
            # Near zero rotation
            axis = self.q[1:4]
        else:
            axis = self.q[1:4] / s
        
        return angle * axis
    
    def from_axis_angle(self, axis_angle):
        """Create quaternion from axis-angle representation"""
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-10:
            # Small angle approximation
            self.q[0] = 1.0
            self.q[1:4] = axis_angle / 2.0
        else:
            axis = axis_angle / angle
            self.q[0] = np.cos(angle / 2.0)
            self.q[1:4] = axis * np.sin(angle / 2.0)
        
        self.normalize()
        return self
    
    def euler_angles(self):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        Using ZYX convention
        """
        w, x, y, z = self.q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def to_rotation_matrix(self):
        """Convert quaternion to 3x3 rotation matrix"""
        w, x, y, z = self.q
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    @staticmethod
    def from_rotation_matrix(R):
        """Create quaternion from rotation matrix"""
        q = Quaternion()
        
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q.q[0] = 0.25 / s
            q.q[1] = (R[2, 1] - R[1, 2]) * s
            q.q[2] = (R[0, 2] - R[2, 0]) * s
            q.q[3] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q.q[0] = (R[2, 1] - R[1, 2]) / s
                q.q[1] = 0.25 * s
                q.q[2] = (R[0, 1] + R[1, 0]) / s
                q.q[3] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q.q[0] = (R[0, 2] - R[2, 0]) / s
                q.q[1] = (R[0, 1] + R[1, 0]) / s
                q.q[2] = 0.25 * s
                q.q[3] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q.q[0] = (R[1, 0] - R[0, 1]) / s
                q.q[1] = (R[0, 2] + R[2, 0]) / s
                q.q[2] = (R[1, 2] + R[2, 1]) / s
                q.q[3] = 0.25 * s
        
        q.normalize()
        return q