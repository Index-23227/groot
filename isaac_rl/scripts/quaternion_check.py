from scipy.spatial.transform import Rotation

# T derived from your calibration session — save this permanently
r_in  = Rotation.from_quat([0.3536, 0.3536, 0.6124, 0.6124])   # x,y,z,w
r_out = Rotation.from_quat([0.6124, 0.3536, -0.3536, 0.6124])
T = r_out * r_in.inv()

def euler_to_cfg_quat(x_deg, y_deg, z_deg):
    desired = Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True)
    q = (T.inv() * desired).as_quat()  # x,y,z,w
    return (q[3], q[0], q[1], q[2])    # w,x,y,z for Isaac Lab

print(euler_to_cfg_quat(176, 52, 93))