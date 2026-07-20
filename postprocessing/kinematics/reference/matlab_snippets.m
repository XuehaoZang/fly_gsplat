%% ===========================================================================
%  reference/matlab_snippets.m   —  REFERENCE ONLY, NOT RUN
%  Source: https://github.com/AbbyLeung/Fly-Hull-Reconstruction
%  Purpose: authoritative geometric definitions of the OLD voxel/visual-hull
%           method, for comparison while rewriting T4 (point-cloud kinematics).
%
%  These four functions are the "single source of truth" for how the old
%  pipeline defined body/wing angles and, crucially, the wing CHORD.
%  We are NOT porting this code; we reference it to (a) match angle
%  conventions and (b) reproduce find_chords_quad as the BASELINE that the
%  new point-cloud chord method must beat (especially near stroke reversal).
%
%  T4 differences from this MATLAB code (see reference/calc_kinematics.md):
%    - We compute angles in the STROKE-PLANE frame (per the Python notebook),
%      not the lab-horizontal frame used by calcEta below.
%    - Chord: this file finds the "most distant voxel pair" in a thin strip;
%      the new method does segmented, Gaussian-normal-weighted robust fitting.
%  ===========================================================================


%% ############################################################################
%% FILE 1/4 : calc kinematics/calcAnglesRaw_Sam.m
%%   yaw(psi)/pitch(beta) via atan2/asin on AHat; wing phi/theta via span;
%%   body-frame transform with hard-coded thetaB0 = 45 deg stroke-plane tilt.
%% ############################################################################
%--------------------------------------------------------------------------
% function to calculate body and wing angles from the data structure output
% by hullAnalysis. Based on calcAngles_quick_and_dirty_mk*.m
%
% this version uses a vector-based calculation of the wing pitch angles eta.
% this calculation is done in the function calcEta.m :
% calcEta(s, c, leftRight)
%
%
% each matrix contains 8 columns for
% beta (body), psi (body), phiR, phiL, thetaR, thetaL, etaR, etaL
%
% subfunction used:
% calcBodyEulerAngles
% calcEta
%--------------------------------------------------------------------------
function [anglesLabFrame, anglesBodyFrame, t, newEtaLab, newEtaBody, sp_rho,...
    smoothed_rho, rho_t, rho_samp, rotM_YP, rotM_roll, largePertFlag] = ...
    calcAnglesRaw_Sam(data, plotFlag,largePertFlag)
%--------------------------------------------------------------------------
%% params and inputs
%
if (~exist('plotFlag','var'))
    plotFlag = false ;
end
if (~exist('largePertFlag','var'))
    largePertFlag = false ;
end
% when putting the body in consistent reference frame, make the body axis
% pitched up by 45 degrees from the horizontal plane. this should make the
% stroke plane roughly horizontal in the transformed frame.
%
% NB: should I try to get a better estimate for this? in principle, it
% would affect stroke vs deviation wing angles
thetaB0 = 45 ;
thetaB0rad = thetaB0 * pi / 180 ;

% misc useful params
DEG2RAD = pi / 180 ;
RAD2DEG   = 180 / pi ;
tol = 1e-12 ;
fps = data.params.fps ;

% the angles will ultimately be stored in Nx8 or Nx9 matrices--the script
% below defines variables whose names are the greek letters corresponding
% to the angles, so it's easier to access the right angle
defineConstantsScript

%--------------------------------------------------------------------------
%% initialize data containers
anglesLabFrame  = zeros(data.Nimages, 9);
anglesBodyFrame = zeros(data.Nimages, 8);

newEtaLab  = zeros(data.Nimages,2) ;
newEtaBody = zeros(data.Nimages,2) ;

% timing info
if (isfield(data,'startAnalysisTimeMS'))
    startTime = data.startAnalysisTimeMS * fps / 1000 ;
    endTime = data.endAnalysisTimeMS * fps / 1000 ;
    Np = endTime - startTime + 1 ;
else
    startTime = data.params.startTrackingTime ;
    endTime   = data.params.endTrackingTime ;
    Np = data.Nimages ;
end

t = (startTime:endTime) / fps  ; % in SEC
startFrame = startTime - data.params.startTrackingTime + 1 ;
endFrame   = startFrame + Np - 1 ;
frameIndices = startFrame:endFrame ;


allT   = (0:data.Nimages-1) + data.params.startTrackingTime ;
allT   = allT / data.params.fps ; % in SEC
% -------------------------------------------------------------------------
%% calculate raw psi (yaw) and raw beta (pitch) for all frames (IN DEGREES)
% ----------------------------------------------------------
if largePertFlag
    % in cases where gimbal lock may be a problem, use a frame-by_frame
    % estimate for the body pitch and yaw angles
    [rawBeta, rawPsi, rotM_YP, largePertFlag] = ...
        calcPitchLargePert(data, plotFlag) ;
else
    % otherwise proceed as normal
    rawPsi  = zeros(data.Nimages,1) ; % yaw
    rawBeta = zeros(data.Nimages,1) ; % pitch
    rotM_YP = zeros(3,3,data.Nimages) ; % rotation matrices that will unyaw + unpitch
    for k=1:data.Nimages
        AHat = data.AHat(k,:) ;
        rawPsi(k)  = atan2(AHat(2),AHat(1)); % body angle with respect to x axis 
        rawBeta(k) = asin(AHat(3));  % body angle with respect to the horizon 
        rotM_YP(:,:,k) = eulerRotationMatrix(rawPsi(k), rawBeta(k),0) ; 
    end
    % convert rawPsi and rawBeta to degrees
    rawPsi = RAD2DEG*rawPsi ; 
    rawBeta = RAD2DEG*rawBeta ; 
end
% -------------------------------------------------------------------------
%% calculate body roll
% calc roll angles at data.rhoTimes
if (~isfield(data,'rhoTimes')) || (isempty(data.rhoTimes))
    %[rhoTimes, rollVectors] = estimateRollVectors(data);
    disp('data.rhoTimes is not defined.') ;
    rhoFlag = false ;
else
    rhoTimes    = data.rhoTimes ;
    rollVectors = data.rollVectors ;
    rhoFlag = true ;
end

if rhoFlag && isfield(data, 'newRhoSamp')
    % if possible, uses 'newRhoSamp', estimated in the script for large 
    % perturbation corrections
    [smoothed_rho, sp_rho, rho_t, rho_samp] = calcRollNewRhoSamp(data,t) ; 
elseif rhoFlag && ~isfield(data, 'newRhoSamp')
    % if we have estimates of roll vector, determine body roll angle
    [smoothed_rho, sp_rho, rho_t, rho_samp, rotM_roll] = ...
        calcBodyRoll(rhoTimes, rollVectors, t, rotM_YP, data.params,...
         largePertFlag) ;
else
    smoothed_rho = 0 ; %rho0 ;
    sp_rho = [] ;
    rho_t = allT ;
    rho_samp = 0 ;
end

%--------------------------------------------------------------------------
%% now calculate wing angles.
% to do so, we move everything to the body frame by undoing the yaw, pitch,
% and roll rotations. then calculate wing angles in the new frame. We also
% calculate wing angles in the lab frame
for k=1:data.Nimages
    
    % ------------------
    %% COPY RELEVANT DATA
    % ------------------
    cb = data.bodyCM(k,:)' ;
    cr = data.rightWingCM(k,:)' ;
    cl = data.leftWingCM(k,:)' ;
    
    rightChordHat = data.rightChordHats(k,:)' ;
    leftChordHat  = data.leftChordHats(k,:)' ;
    rightSpanHat  = data.rightSpanHats(k,:)' ;
    leftSpanHat   = data.leftSpanHats(k,:)' ;
    
    % get wing pitch right away
    newEtaLab(k,1) = calcEta(rightSpanHat, rightChordHat,'right') ;
    newEtaLab(k,2) = calcEta(leftSpanHat, leftChordHat,'left') ;
    
    etaRdeg = newEtaLab(k,1)  ;
    etaLdeg = newEtaLab(k,2)  ;
    
    % Use the smoothed version of the roll angle rho
    bodyRollAngle = smoothed_rho(k) ;
    
    % ---------------------------------------------------------------------
    %% CALCULATE ANGLES IN THE LAB FRAME
    % ---------------------------------------------------------------------
    % body
    psiDeg  = rawPsi(k) ;
    betaDeg = rawBeta(k) ;
    
    % right wing
    phiRdeg   = atan2(rightSpanHat(2),rightSpanHat(1)) * RAD2DEG ;
    thetaRdeg = asin(rightSpanHat(3)) * RAD2DEG;
    
    % left wing
    phiLdeg   = atan2(leftSpanHat(2), leftSpanHat(1)) * RAD2DEG;
    thetaLdeg = asin(leftSpanHat(3)) * RAD2DEG;
    
    % store result
    anglesLabFrame(k,:) = ...
        [psiDeg betaDeg phiRdeg thetaRdeg etaRdeg phiLdeg...
        thetaLdeg etaLdeg bodyRollAngle] ;
    
    % -----------------------------------------------
    %% CALCULATE ANGLES IN THE BODY FRAME OF REFERENCE
    % -----------------------------------------------
    
    % first rotation matrix bring (xlab, ylab, zlab) to
    % (xbody ybody zbody) such that xbody is AHat
    
%     % convert to radians
%     phiB   = psiDeg*deg2rad ;
%     thetaB = betaDeg*deg2rad ;
%     psiB   = bodyRollAngle*deg2rad ;
%     
%     % rotation matrix to strict body axis
%     M1 = eulerRotationMatrix(phiB,thetaB,psiB ) ;
    
    % rotation matrix to strict body axis
    M1 = squeeze(rotM_roll(:,:,k)) * squeeze(rotM_YP(:,:,k)) ; 
    
    % pitch down by thetaB w.r.t body axis
    M2 = eulerRotationMatrix(0, -thetaB0rad, 0) ;
    M = M2 * M1 ;
    
    % M1' rotates a vector by (phi, theta, psi) in the lab frame, e.g.
    % M1' * [1;0;0] = data.AHat(k,:)'
    
    % M1  undoes the rotation of M1'. alternative description is that
    % M1 * data.AHat(k,:)' = [1;0;0]
    % give the coordinate of a lab-frame vector as described in the body frame
    
    % M2' * [1;0;0] = unit vector pitched down by thetab0
    % M2 * v = gived v pitched up by thetab0
    
    % first, represent the vector in the body-bound frame coordinates
    % then perform the rotation about the body roll axis
    % then represent back in whatever frame you want?
    
    % ---------------
    %% RIGHT WING
    % ---------------
    rotCR = M * (cr - cb) ; % rotated right wing center of mass
    rotSpanR = M * rightSpanHat ; % rotated right span
    rotChordR = M * rightChordHat ; % rotated right chord
    
    % get angles
    phiRdeg   = unwrap(atan2(rotSpanR(2),rotSpanR(1))) * RAD2DEG;
    thetaRdeg = asin(rotSpanR(3)) * RAD2DEG ;
    newEtaBody(k,1) = calcEta(rotSpanR, rotChordR,'right') ;
    etaRdeg = newEtaBody(k,1) ;
    % ---------------
    %% LEFT WING
    % ---------------
    rotCL = M * (cl - cb) ; % rotated left wing center of mass
    rotSpanL = M * leftSpanHat ; % rotated left span
    rotChordL = M * leftChordHat ; % rotated left chord
    
    % get angles
    phiLdeg   = unwrap(atan2(rotSpanL(2),rotSpanL(1))) * RAD2DEG;
    thetaLdeg = asin(rotSpanL(3)) * RAD2DEG ;
    newEtaBody(k,2) = calcEta(rotSpanL, rotChordL,'left') ;
    etaLdeg = newEtaBody(k,2) ;
    
    %------------------------------------------
    % store result
    anglesBodyFrame(k,:) = ...
        [0 0 phiRdeg thetaRdeg etaRdeg phiLdeg thetaLdeg etaLdeg] ;
    
    
end

%--------------------------------------------------------------------------
%% store angles in matrices

anglesBodyFrame(:,PHIR)   = + anglesBodyFrame(:,PHIR) ; % NOTE MINUS

anglesLabFrame  = unwrap(anglesLabFrame/RAD2DEG) * RAD2DEG ;
anglesBodyFrame = unwrap(anglesBodyFrame/RAD2DEG) * RAD2DEG ;

%--------------------------------------------------------------------------
%% plot results?
labels = {'\psi','beta','\phi_R','\theta_R', '\eta_R',...
    '\phi_L','\theta_L','\eta_L','\rho'} ;

plotReorder = [ PSI BETA PHIL PHIR ETAL ETAR THETAL THETAR ];
if (plotFlag)
    figure('name','Angles body frame') ;
    for s=1:8
        subplot(4,2,s) ;
        ind = plotReorder(s) ;
        plot(allT, anglesBodyFrame(:,ind),'ko-') ;
        xlabel('Time [ms]') ;
        title([labels{ind} ' body frame']) ;
        grid on ; box on ;
        set(gca,'xlim',[t(1) t(end)]);
    end
    figure('name','Angles lab frame') ;
    for s=1:8
        subplot(4,2,s) ;
        ind = plotReorder(s) ;
        plot(allT, anglesLabFrame(:,ind),'o-','color',[0 0.8 0]) ;
        xlabel('Time [ms]') ;
        title([labels{ind} ' lab frame']) ;
        grid on ; box on ;
        set(gca,'xlim',[allT(1) allT(end)]);
    end
    
end

end

%% ############################################################################
%% FILE 2/4 : calc kinematics/calcEta.m
%%   Wing pitch eta (pronation/supination). NOTE: reference direction phihat
%%   is defined against the LAB vertical (z-hat). T4 uses a stroke-plane ref.
%% ############################################################################

function eta = calcEta(s, c, leftRight, plotFlag)
% calculates the wing pitch angle eta in degrees
% input parameters:
% s - span unit vector
% c - chord unit vector
% leftRight - a string that indicates whether it's the right or left wing.
% only the lower cae of the first character is considered ('l' or 'r')

if (size(s,1)==1)
    s = s' ;
end
if (size(c,1)==1)
    c = c' ;
end

if (nargin==3)
    plotFlag = false ;
end
delta    = 1000*eps ;
% -----------------------
% determine left or right
% -----------------------

chr = lower(leftRight(1)) ;

switch(chr)
    case 'r'
        wingFlag = 1 ;
    case 'l'
        wingFlag = -1 ;
    otherwise
        disp('illegal value for the leftRight input argument') ;
        eta = NaN ;
        return
end


% -----------------------------
% calculate eta - using vectors
% -----------------------------

phihat = - cross(s, [0 ; 0 ; 1]) * wingFlag ;
phihat = phihat / norm(phihat) ;

eta = acos( dot(c, phihat) ) * 180 / pi ;

% take care of eta's sign
if (eta~=0)
    v3 = cross(phihat, c) ; % v3 should be parallel to s
    v3 = v3 / norm(v3) ;
    % if v3 and s are on the same direction, then sgn=+1
    % if v3 and s are on opposite direction then sgn=-1
    
    sgn = dot(v3, s) ; % sgn is either 1 or -1
    sgn2 = round(sgn) ; % to prevent cases where sgn = +-1 +1 epsilon
    if (abs(sgn-sgn2)>delta)
        disp('ERR in the calculation of eta. Check this.') ;
        %keyboard ;
    end
    
    eta = eta * sgn2 * wingFlag ;
end
% print the result
% disp(['method 1: eta=' num2str(eta) ]) ;

if (eta<0) 
    eta = eta + 360 ;
end

% ----
% plot
% ----
if (plotFlag)
    v1 = - cross(s,[0 ;0 ;1]) * wingFlag ; % projection of c onto the xy plane
    v1 = v1 / norm(v1) ;                   % normalize
    v2 = - cross(v1, s) * wingFlag ;
    origin = [ 0 ; 0 ; 0 ] ;
    
    figure ;
    hold on ;
    myplot(origin, origin,'ks') ;
    myplot(origin, s, 'ro-') ;
    %{
    myplot(s, c, 'bo-') ;
    myplot(s, v1,'ko--') ;
    myplot(s, v2,'ks--') ;
    myplot(s, phihat,'gd-') ;
    %}
    myplot(origin, c, 'bo-') ;
    myplot(origin, v1,'ko--') ;
    myplot(origin, v2,'ks--') ;
    myplot(origin, phihat,'gd-') ;
    hold off ;
    axis equal ;
    grid on ;
    box on ;
    view(3) ;
    xlabel('x') ; ylabel('y') ; zlabel('z') ;
end
end

function myplot(v1,v2,colstr)
v3 = v1+v2 ;
plot3( [v1(1) v3(1)], [v1(2) v3(2)], [v1(3) v3(3)],colstr,'linewidth',2) ;
end

%% ############################################################################
%% FILE 3/4 : core functions/estimate_wing_vecs.m
%%   How span & chord & wingtip were extracted from wing voxels.
%% ############################################################################
%--------------------------------------------------------------------------
% Function to estimate the span and chord vectors for a wing. Based on the
% methods from hullAnalysis
%
%   INPUTS:
%       -wing_vox = voxel coordinates for the wing
%
%       -ref_vecs = reference vectors to determine direction of span and
%       chord for left vs right wing. It is assumed
%       that the reference vectors in this case are a list of [bodyCM(i);
%       bodyCM(i-1); wingTip(i-1)].
%
%       -wingLength = estimate of wing length. comes from data struct as
%           wingLength = 35 * params.pixPerCM / 232 ;
%
%       -span_ref (optional) = reference span against which to compare
%       -chord_ref (optional) = reference chord against which to compare
%
%   OUTPUTS:
%       -span = estimate of span vector
%       -chord = estimate of chord vector
%       -chord_alt = estimate of alternate chord vector
%       -N_vox = number of voxels in wing
%--------------------------------------------------------------------------
function [spanHat, chordHat, chordAltHat, Nvox, wingTip, diag1, diag2] = ...
    estimate_wing_vecs(wing_vox, ref_vecs, wingLength, span_ref, ...
    chord_ref, wingCM)
%--------------------------------------------------------------------------
%% deal with function inputs
if ~exist('span_ref','var') 
    span_ref = [] ;
end
if ~exist('chord_ref','var')
    chord_ref = [] ;
end
if ~exist('wingCM','var')
    wingCM = mean(wing_vox) ; 
end
%--------------------------------------------------------------------------
%% params
LL = wingLength * 0.55 ; % used with farthestPoint
Nvox = size(wing_vox,1) ;
debugFlag = false ;

bodyCM = ref_vecs(1,:) ;
bodyCM_prev = ref_vecs(2,:) ;
wingTip_prev = ref_vecs(3,:) ;

%--------------------------------------------------------------------------
%% get span
if isempty(span_ref) || any(isnan(span_ref))
    % first pass: span = vector from body center of mass to distal wing tip
    farPoint    = farthestPoint(wing_vox, bodyCM, LL) ;
    spanHat = farPoint - wingCM ;
    spanHat = spanHat' ;
    spanHat = spanHat / norm(spanHat) ;
    
    % makes wing span point outward
    if dot(wingCM-bodyCM,spanHat) < 0
        spanHat = -spanHat;
    end
    
    wingTip = findWingTip(wing_vox, spanHat', wingCM);
    if (isnan(wingTip(1)))
        wingTip = farPoint ;
    end
    
    % second pass: recalculate span vector based on the refined wing tip
    spanHat = wingTip - wingCM ;
    spanHat = spanHat' ;
    spanHat = spanHat / norm(spanHat) ;
else
    pca_coeffs = pca(wing_vox);
    spanHat = pca_coeffs(:,1);
    
    if dot(spanHat,span_ref) <= 0 
        spanHat = -spanHat;
    end
    wingTip = findWingTip(wing_vox, spanHat', wingCM);
end
%----------------------------------------------------------------------
%% find chord vector
[chordHat,chordAltHat, diag1, diag2] = find_chords_quad(wing_vox, spanHat', ...
    wingTip, wingTip_prev, bodyCM, bodyCM_prev) ;

if ~isempty(chord_ref)
    if compare_vectors(chordHat, chord_ref) > ...
            compare_vectors(chordAltHat, chord_ref)
        tmp = chordHat ;
        chordHat = chordAltHat ;
        chordAltHat = tmp ;
    end
end
%----------------------------------------------------------------------
%% plot results?
if debugFlag
    figure ;
    hold on
    plot3(wing_vox(:,1), wing_vox(:,2), wing_vox(:,3),'k.')
    plot3((wingCM(1) + 24*[0, spanHat(1)]), (wingCM(2) + 24*[0 spanHat(2)]),...
        (wingCM(3) + 24*[0 spanHat(3)]),'b-','linewidth',4)
    plot3((wingCM(1) + 12*[0 chordHat(1)]), (wingCM(2) + 12*[0 chordHat(2)]),...
        (wingCM(3) + 12*[0 chordHat(3)]),'r-','linewidth',4)
    plot3((wingCM(1) + 12*[0 chordAltHat(1)]), (wingCM(2) + 12*[0 chordAltHat(2)]),...
        (wingCM(3) + 12*[0 chordAltHat(3)]),'r:','linewidth',4)
    legend({'Data','Span','Chord','Alt. Chord'})
    axis equal
    grid on
    
end


end

%% ############################################################################
%% FILE 4/4 : core functions/find_chords_quad.m
%%   *** BASELINE for T4-S4 ***  Chord = most-distant voxel pair inside a thin
%%   mid-span strip, disambiguated by wingtip velocity. This is exactly the
%%   fragile step that blurs near stroke reversal; the new method replaces it.
%% ############################################################################
%--------------------------------------------------------------------------
% re-calculate the chord vector
%--------------------------------------------------------------------------
function [chordHat,chordAltHat, diag1, diag2] = ...
    find_chords_quad(WingVoxels, span, WingTip, WingTip_prev, body_COM,...
                        prev_body_COM)
%--------------------------------------------------------------------------
%% params
wingTipVelocityThreshold = 5 ;
chordFraction     = 0.33 ; % fraction of the chord voxels used to find chord
delta             = 2.0;  % strip width used in finding the wing chord

%--------------------------------------------------------------------------
%% find the voxels that are not far from wing CM and ~perpendicular to span
Nvox = size(WingVoxels,1) ;
mat1 = WingVoxels - repmat(mean(WingVoxels),Nvox,1) ; 
mat2 = repmat(span, Nvox, 1) ;

distFromMidSpan = abs(sum(mat1.*mat2,2) ) ;
clear mat1 mat2

% the two points that are farthest apart within this set of voxels defines
% the chord
chordRowsInd = find(distFromMidSpan<delta) ;

if (numel(chordRowsInd) < 5)
    % first try a larger delta
    chordRowsInd = find(distFromMidSpan<3*delta) ;
    % check if still empty
    if (numel(chordRowsInd) < 5)
        %error('hullAnalysis:Chord','Bad clustering - empty right chord') ;
        disp('Error: empty chord')
        chordHat = [nan, nan, nan]  ; 
        chordAltHat = [nan, nan, nan] ;
        diag1 = nan ; 
        diag2 = nan ; 
        return
    end
end
clear distFromMidSpan
Nvox    = length(chordRowsInd);
sqdist = (WingVoxels(chordRowsInd,:) - repmat(mean(WingVoxels),Nvox,1)).^2 ;
distVec = (sum(sqdist,2)).^0.5 ;
clear sqdist

% select only the top quarter of the voxels, i.e the most distant from
% wing centroid

[~, sortedInd] = sort(distVec,'descend') ;
selectedInd    = chordRowsInd(sortedInd(1:ceil(Nvox*chordFraction))) ;
%selectedIndRight = selectedInd ;
clear distVec

% find the most distant pair
distMat = squareform (pdist (WingVoxels(selectedInd,:))) ;

[maxRowVec, Irow] = max(distMat,[],1) ;
[~, Icol] = max(maxRowVec) ;
Irow = Irow(Icol) ;

if (distMat(Irow, Icol)~=max(distMat(:)))
    disp('Error with finding max. plz check.') ;
    disp('problem 6?') ;
    %keyboard ;
end

%--------------------------------------------------------------------------
%% define the chord vector
% the following indices give voxel coordinate:
vox1Ind = selectedInd(Irow) ; 
vox2Ind = selectedInd(Icol) ; 

chordHat = WingVoxels(vox1Ind,:)' - WingVoxels(vox2Ind,:)' ;
% force chord to be vertical to the span vector
chordHat = chordHat - span.' * dot(span, chordHat) ;
diag1 = norm(chordHat) ; % will be used later
chordHat = chordHat / norm(chordHat) ;

if (chordHat(3)<0)
    chordHat = - chordHat ;
end

%--------------------------------------------------------------------------
%% find second diagonal of wing parallelogram
mat1 = WingVoxels(chordRowsInd,:) - repmat(mean(WingVoxels),Nvox,1) ; 
% calc the vector normal to the span and chord
wingNormVec = cross(span, chordHat);
% calc the (signed) distance from each point in mat1 to the wing plane
mat2 = repmat(wingNormVec, Nvox, 1) ;
distVec = sum(mat1.*mat2,2) ;

% find the largest positive and largest negative distances, which
% correspond to the farthest voxels on each side of the plane

[maxval, indmax] = max(distVec) ; % index into rightWingVoxels
[minval, indmin] = min(distVec) ; % index into rightWingVoxels

if (maxval<=0 || minval>=0)
    disp('error in finding alternative chord vector for wing') ;
    %keyboard ;
end

indmin = chordRowsInd(indmin) ;
indmax = chordRowsInd(indmax) ;
chordAltHat = WingVoxels(indmax,:)' - WingVoxels(indmin,:)' ;

% force alternative chord to be vertical to the span vector
chordAltHat = chordAltHat - span.' * dot(span, chordAltHat) ;

diag2 = norm(chordAltHat) ; % will be used later
% now normalize
chordAltHat = chordAltHat / norm(chordAltHat) ;

% choose the sign of the alternative chord vector such that it is
% has positive overlap with the "main" chord vector

%if ( dot(chord1AltHat, chord1Hat) < 0 )
%    chord1AltHat = - chord1AltHat ;
%end

if (chordAltHat(3)<0)
    chordAltHat = - chordAltHat ;
end

% decide whether to swap the "main" and "alternative" chord vectors
% if one of the diagonals is siginficantly longer, choose the longer
% one and do not proceed to the velocity criterion below
diagSwapFlag     = false ;
velocitySwapFlag = false ;

if (diag2/diag1 >= 1.3)
    diagSwapFlag = true ;
    %contProcess = false ;
    %disp('swap based on large ratio') ;
end

%% use wing tip 'velocity' wrt the body to refine chord estimate
% previous version calculate wing centroid velocity:
if all(isnan(WingTip_prev))
    WingTip_prev = WingTip ; 
end
vWing =  ( WingTip - body_COM )-(WingTip_prev - prev_body_COM) ;
% keep only the component perpendicular to the span vector
vWing = vWing - span * dot(span.', vWing) ;
nrm = norm(vWing) ;

% check to see if chord points in direction of wing velocity
if (nrm~=0)
    vWing = vWing / nrm ;
    dot1 = dot(chordHat, vWing) ;
    dot2 = dot(chordAltHat, vWing) ;
    %disp(['dot1=' num2str(dot1) '  dot2=' num2str(dot2)]) ;
    if (dot2>dot1) % swap
        velocitySwapFlag = true ;
    end
    if (dot1<0 && dot2<0 && nrm>=wingTipVelocityThreshold && ~velocitySwapFlag)
        %disp('--> inverting right chord. not swapping.') ;
        chordHat = - chordHat ;
    end
end

swapFlag = (velocitySwapFlag && nrm>=wingTipVelocityThreshold) || ... % believe velocity if |v|>2
    (diagSwapFlag && nrm<wingTipVelocityThreshold) ;

% probably need a smarter way to "weigh" the two types of swaps
if (swapFlag)
    tmp = chordHat ;
    chordHat = chordAltHat ;
    chordAltHat = tmp ;

    tmp = diag1 ;
    diag1 = diag2;
    diag2 = tmp ;

    clear tmp ;
    %disp('Swapped right chord with alternative chord') ;
end

if chordHat(3)<0
    chordHat=-chordHat;
end
end