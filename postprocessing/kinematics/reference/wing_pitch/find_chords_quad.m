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