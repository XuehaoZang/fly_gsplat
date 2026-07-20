# =============================================================================
# reference/python_snippets.py   —  REFERENCE ONLY, NOT RUN, CONTAINS KNOWN BUGS
# Source: run_pose_estimation.ipynb (cells 1-3), the messy Python port.
#
# Purpose: authoritative source for the STROKE-PLANE-relative angle
# definitions that T4 adopts (phi / theta / eta and the roll formula).
# We reference these to reproduce the geometry exactly and to cross-check
# the rewritten functions numerically. We DO NOT run or import this file.
#
# ADOPTED from here (geometry is correct):
#   - cell 2  calculate_roll(...)            -> roll definition (authoritative)
#   - cell 3  project_on_plane / calculate_phi -> phi, theta, eta in SP frame
#
# REPLACED in T4 (do NOT copy these hacks):
#   - cell 3  `psi[psi < -100] = -psi[...]`  : unprincipled sign patch for the
#             chord; T4 fixes the sign via physical LE->TE ordering instead.
#   - cell 1  stroke-plane bootstrap: SP normal is fit from MULTI-FRAME wingtip
#             clouds, then ybody2 = cross(xbody, SP_normal), then
#             sp_normal = rodrigues(xbody, ybody2, -45deg). T4 is SINGLE-FRAME:
#             ybody comes from the R->L wing-root (hinge) line; SP normal is
#             xbody rotated -45deg about ybody. Multi-frame path kept only to
#             show what we replaced (interface left open to accept an external
#             stroke_plane_normal later).
#
# Coordinate / convention notes for T4 (see reference/calc_kinematics.md):
#   - units: meters ; up = +z (from cam4 extrinsics)
#   - L/R are the fly's own left/right ; angles output in stroke-plane frame
# =============================================================================


# ####################################################################
# CELL 1/3 : stroke plane + body y-axis (MULTI-FRAME; T4 replaces this)
#   NOTE: `Plane.best_fit` over +/-delta_frames wingtips = multi-frame.
#   NOTE: rodrigues(..., -45deg) is the hard-coded stroke-plane tilt.
# ####################################################################
from skspatial.objects import Plane, Points
# calculate body axes the same way as in the hull reconstruciton code
area_le = np.array([np.sum(np.cross(frames_fly[frame].right_wing['le_ransac'][1],frames_fly[frame].left_wing['le_ransac'][1])) for frame in range(len(frames_fly))])
sign_le = np.sign(area_le)
diff_sign = sign_le[0:-1] - sign_le[1:]
indices = np.argwhere(diff_sign == -2)
delta_frames = 10       # can change delta frames for taking avg

sp = []
key_y_vectors = []
key_indices = []
for num_of_idx,frame in enumerate(indices[:-1]):
    frame = frame[0]

    min_idx,max_idx = np.max((frame - delta_frames,0)),np.min((frame + delta_frames,len(area_le)))
    tips_for_sp = np.vstack([(frames_fly[frame].right_wing['tip_mean'],frames_fly[frame].left_wing['tip_mean']) for frame in range(min_idx,max_idx)])
    points = Points(tips_for_sp)        # tips for sp
    normal_to_sp = Plane.best_fit(points)
    normal_to_sp = np.array(np.atleast_2d(normal_to_sp.normal))
    sign = 1 if np.dot(normal_to_sp[0],np.array([0,0,1])) > 0 else -1

    
    frames_fly[frame].stroke_plane = sign*normal_to_sp
    frames_fly[frame].ybody2 = np.cross(frames_fly[frame].xbody,frames_fly[frame].stroke_plane)[0]
    projected_tip = np.dot(frames_fly[frame].right_wing['tip_mean'],frames_fly[frame].ybody2)
    projected_body_cm = np.dot(frames_fly[frame].body_cm,frames_fly[frame].ybody2)

    if projected_tip > projected_body_cm:
        frames_fly[frame].ybody2 = -frames_fly[frame].ybody2

    if np.dot( frames_fly[frame].xbody,np.array([0,0,1])) < 0:
        frames_fly[frame].xbody = -frames_fly[frame].xbody


    frames_fly[frame].zbody2 = np.cross(frames_fly[frame].ybody2,frames_fly[frame].xbody)
    
    sp.append(frames_fly[frame].stroke_plane[0])
    key_indices.append(frame)
    key_y_vectors.append(frames_fly[frame].ybody2)

plt.plot(area_le,'*', label='All Area LE')
plt.plot(indices,area_le[indices],'*',label='Key Indices')

from scipy.interpolate import CubicSpline

all_frames_indices = np.arange(len(frames_fly))

cs = CubicSpline(np.hstack(key_indices), key_y_vectors, axis=0, extrapolate=True)
y_interpolated = cs(all_frames_indices)
for i in range(len(frames_fly)):
        x_curr = frames_fly[i].xbody
        y_smooth = y_interpolated[i]
        

        if np.dot( frames_fly[i].xbody,np.array([0,0,1])) < 0:
            frames_fly[i].xbody = -frames_fly[i].xbody

        
        # A. Orthogonalize (Gram-Schmidt)
        # The interpolation might drift slightly off 90 degrees from X.
        # Remove the component of Y that is parallel to X.
        proj = np.dot(y_smooth, x_curr)
        y_final = y_smooth - (proj * x_curr)
        
        # Normalize
        y_final = y_final / np.linalg.norm(y_final)
        
        # B. Recalculate Z
        z_final = np.cross(x_curr, y_final)
        
        # C. Update Frame
        frames_fly[i].ybody2 = y_final
        frames_fly[i].zbody2 = z_final
        
        # (Optional) Update stroke plane normal based on new axes
        # Roughly speaking, Z is the new normal
        frames_fly[i].sp_normal = frames_fly[i].rodrigues_rotate_vector(frames_fly[i].xbody, frames_fly[i].ybody2, -45*np.pi/180)
        # frames_fly[i].stroke_plane = z_final

plt.plot(y_interpolated,label=['Y_interp_X', 'Y_interp_Y', 'Y_interp_Z'])
plt.legend()


# ####################################################################
# CELL 2/3 : yaw / pitch / roll   (roll def is AUTHORITATIVE for T4)
# ####################################################################
from scipy.spatial.transform import Rotation as Rscipy
# calculate roll angle (the same way as the gull reconstruction method)
def calculate_roll(yaw_rad, pitch_rad, ybody_vecs):
    """
    Vectorized roll calculation.
    
    Parameters:
    yaw_rad   : (N,) array of yaw angles
    pitch_rad : (N,) array of pitch angles
    ybody_vecs: (N, 3) array of body Y-vectors
    """
    
    # Pre-compute trig functions
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    
    # 1. Construct Intermediate Axes (Zero-Roll Frame)
    
    # ey: The Node Vector (Horizontal, perpendicular to yaw)
    # Shape: (N, 3) using stack or column_stack
    ey = np.column_stack([-sy, cy, np.zeros_like(sy)])
    
    # ez: The Intermediate Z Axis
    # CORRECTION: Note the negative signs on x and y components.
    # Pitch Up -> Z tilts Back (negative X/Y direction)
    ez = np.column_stack([-sp*cy, -sp*sy, cp])

    # 2. Project actual body Y onto these intermediate axes
    # We use (A * B).sum(axis=1) as a vectorized dot product
    Yy = np.sum(ybody_vecs * ey, axis=1)
    Yz = np.sum(ybody_vecs * ez, axis=1)

    # 3. Calculate Roll
    return np.arctan2(Yz, Yy)


xbody_mat = np.vstack([frame.xbody for frame in frames_fly])
z_lab = xbody_mat*0 + [0,0,1]

pitch = 90-np.arccos(np.sum(xbody_mat*z_lab,axis = 1))*180/np.pi
yaw = np.arctan2(xbody_mat[:,1],xbody_mat[:,0])*180/np.pi

yaw_rad = yaw * np.pi / 180
pitch_rad = pitch * np.pi / 180
ybody_array = np.vstack([frame.ybody2 for frame in frames_fly])
roll = np.unwrap(calculate_roll(yaw_rad, pitch_rad, ybody_array))*180/np.pi-180


# roll = np.unwrap([calculate_roll(yaw,pitch,0,frame.ybody2) for yaw,pitch,frame in zip(yaw*np.pi/180,pitch*np.pi/180,frames_fly[200:2500])])


fig, ax = plt.subplots(3,1)
ax[0].plot(yaw, label='yaw')
ax[1].plot(pitch, label='pitch')
ax[2].plot(roll, label='roll')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
ax[2].legend(loc='upper right')


# ####################################################################
# CELL 3/3 : wing phi / theta / eta in the STROKE-PLANE frame
#   ADOPT: project_on_plane, calculate_phi (phi/theta/eta geometry)
#   REJECT: the `psi[psi < -100]` sign patch near the end
# ####################################################################
# calculate wing angles

def project_on_plane( normal, vector):
    projected_vector = vector - np.atleast_2d(np.sum(normal*vector,axis = 1)).T*normal
    return projected_vector / np.atleast_2d(np.linalg.norm(projected_vector, axis = 1)).T


# phi


def calculate_body_vectors(frames_fly):

    sp_normal = np.vstack([frame.sp_normal for frame in frames_fly])
    xbody_mat = np.vstack([frame.xbody for frame in frames_fly])
    ybody_mat = np.vstack([frame.ybody2 for frame in frames_fly])

    xbody_on_sp = project_on_plane(sp_normal,xbody_mat)
    ybody_on_sp = project_on_plane(sp_normal,ybody_mat)
    ybody_on_sp = ybody_on_sp / np.atleast_2d(np.linalg.norm(ybody_on_sp, axis = 1)).T
    return xbody_on_sp,ybody_on_sp,sp_normal


def calculate_phi(frames_fly, left,xbody_on_sp,ybody_on_sp,sp_normal):
    if left == 1:
        sign_left = -1
        le_ransac_mat = [frame.left_wing['le_ransac'][1] for frame in frames_fly]
        chord_mat = np.vstack([frame.left_wing['chord'] for frame in frames_fly])
        le_sp_normal = np.cross(sp_normal, le_ransac_mat)

    else:
        sign_left = 1
        le_ransac_mat = [frame.right_wing['le_ransac'][1] for frame in frames_fly]
        chord_mat = np.vstack([frame.right_wing['chord'] for frame in frames_fly])
        le_sp_normal = np.cross(le_ransac_mat,sp_normal)


    le_on_sp = project_on_plane(sp_normal,le_ransac_mat)
    xle = np.sum(le_on_sp*xbody_on_sp,axis = 1)
    yle = np.sum(le_on_sp*ybody_on_sp,axis = 1)
    # Calculate in radians first
    phi_rad = np.arctan2(sign_left*yle, xle)
    
    # "Stitch" the jumps together
    phi_unwrapped = np.unwrap(phi_rad)
    
    # Convert to degrees
    phi = phi_unwrapped * 180 / np.pi
    theta = 90 - np.arccos(np.sum( sp_normal*le_ransac_mat, axis = 1))*180/np.pi

    sp_chord = np.cross(le_ransac_mat,le_sp_normal)
    sp_chord = sp_chord / np.atleast_2d(np.linalg.norm(sp_chord, axis = 1)).T



    ypsi = np.sum(chord_mat*sp_chord, axis = 1)
    xpsi = np.sum(chord_mat*le_sp_normal, axis = 1)
    psi = np.arctan2(sign_left*ypsi,xpsi)*180/np.pi
    psi[psi < -100] = -psi[psi < -100]
    return phi,theta,psi



left = 1
xbody_on_sp,ybody_on_sp,sp_normal = calculate_body_vectors(frames_fly)
phi_left = calculate_phi(frames_fly, left,xbody_on_sp,ybody_on_sp,sp_normal)

fig,ax = plt.subplots(3,1)
ax[0].plot(phi_left[0],'*')
ax[1].plot(phi_left[1],'*')
ax[2].plot(phi_left[2],'*')

left = 0
phi_right = calculate_phi(frames_fly, left,xbody_on_sp,ybody_on_sp,sp_normal)
ax[0].plot(phi_right[0],'*')
ax[1].plot(phi_right[1],'*')
ax[2].plot(phi_right[2],'*')

