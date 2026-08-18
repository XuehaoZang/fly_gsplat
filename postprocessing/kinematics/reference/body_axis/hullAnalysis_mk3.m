
function data = hullAnalysis_mk3 (bodyHull, wing1Hull, wing2Hull, params, ...
    mergedWingsFlag, oneWingEmptyFlags, outputFilename, plotFlag,...
    saveFigFlag, saveFigPath, verboseFlag, saveTempDataFlag)
% data =
%                      Nimages: 464                     num of images
%                          res: [12473860x4 int16]      voxel coordinates of the entire fly + wings
%                       RESIDX: [12473860x3 logical]    indices showing which voxels in res belong to which body part see 3 indices below:
%                      bodyInd: 1
%                 rightWingInd: 2
%                  leftWingInd: 3
%                       bodyCM: [464x3 double]      body center of mass position (all coordinates are in voxels)
%                  rightWingCM: [464x3 double]      right wing cm position
%                   leftWingCM: [464x3 double]      left wing cm position
%               rightChordHats: [464x3 double]      unit vector for right chord
%                leftChordHats: [464x3 double]      unit vector for left chord
%                rightSpanHats: [464x3 double]      unit vector for right span
%                 leftSpanHats: [464x3 double]      unit vector for left span
%                         AHat: [464x3 double]      unit vector for body axis (see also findBodyAxis.m for the refined calculation
%                chord1AltHats: [464x3 double]      irrelevant
%                chord2AltHats: [464x3 double]      irrelevant
%                  rollVectors: [464x3 double]      roll vectory (empty now
%                          pin: [1x1 struct]        irrelevant
%                       params: [1x1 struct]        same params as in input
%                     errorLog: [464x1 logical]     log of errors (0=fine)
%     rightChordTopProjections: [464x1 double]      irrelevant
%      leftChordTopProjections: [464x1 double]      irrelevant
%                   wingLength: 45                  wing length assumed by the algorithm.
%                  diag11Right: [464x1 double]      irrelevant
%                  diag12Right: [464x1 double]      irrelevant
%                   diag21Left: [464x1 double]      irrelevant
%                   diag22Left: [464x1 double]      irrelevant
%                rightWingTips: [464x3 double]      right wing tip pos
%                 leftWingTips: [464x3 double]      left wing tip pos
%                   bodyCM_old: [464x3 double]      body cm from old method
%                     AHat_old: [464x3 double]      body axis from old method


%{
see: http://www.mathworks.com/help/toolbox/stats/bq_679x-24.html
%}
%testing my calcWingLength method:
%[~, wing1Length] = calcWingLength_mk2(wing1Hull) ;
%[~, wing2Length] = calcWingLength_mk2(wing2Hull) ;

%-------------------------------
% Inputs
if ~exist('plotFlag','var') || isempty(plotFlag)
    plotFlag = false ;
end
if ~exist('saveFigFlag','var')
    saveFigFlag = false ;
end
if ~exist('saveFigPath','var')
    saveFigFlag = false ;
    saveFigPath = [] ;
end
if ~exist('verboseFlag','var')
    verboseFlag = false ;
end
if ~exist('saveTempDataFlag','var')
    saveTempDataFlag = false ;
end
%-------------------------------
% params and data
wingLength = 35 * params.pixPerCM / 232 ; %(wing1Length + wing2Length)/2 * params.pixPerCM / 232

noffset = params.startTrackingTime - params.firstTrackableFrame  ;

df = diff(bodyHull(:,1)) ; % [t x y z ]
frameStartIndBodyHull = [1 ; find(df==1)+1] ;
frameEndIndBodyHull   = [frameStartIndBodyHull(2:end)-1 ; size(bodyHull,1)] ;
frameStartIndBodyHull = [zeros(noffset,1) ; frameStartIndBodyHull ] ;
frameEndIndBodyHull   = [zeros(noffset,1) ; frameEndIndBodyHull ] ;

df = diff(wing1Hull(:,1)) ; % [t x y z ]
frameStartIndWing1Hull = [1 ; find(df==1)+1] ;
frameEndIndWing1Hull   = [frameStartIndWing1Hull(2:end)-1 ; size(wing1Hull,1)] ;
frameStartIndWing1Hull = [zeros(noffset,1) ; frameStartIndWing1Hull ] ;
frameEndIndWing1Hull   = [zeros(noffset,1) ; frameEndIndWing1Hull ] ;

df = diff(wing2Hull(:,1)) ; % [t x y z ]
% look for empty wing hulls

frameStartIndWing2Hull = [1 ; find(df==1)+1] ;
frameEndIndWing2Hull   = [frameStartIndWing2Hull(2:end)-1 ; size(wing2Hull,1)] ;
frameStartIndWing2Hull = [zeros(noffset,1) ; frameStartIndWing2Hull ] ;
frameEndIndWing2Hull   = [zeros(noffset,1) ; frameEndIndWing2Hull ] ;

%{
df = diff(res(:,1)) ;
frameStartInd = [1 ; find(df==1)+1] ;
frameEndInd   = [frameStartInd(2:end)-1 ; size(res,1)] ;
frameStartInd = [zeros(noffset,1) ; frameStartInd ] ;
frameEndInd   = [zeros(noffset,1) ; frameEndInd ] ;
%}


clear df noffset;
%disp('not saving figs. press any key') ; pause ;
totalCalcTime     = 0 ;
chordFraction     = 0.33 ; % fraction of the chord voxels used to find chord
delta             = 2.0;  % strip width used in finding the wing chord
tempOffset        = 0;
%probThresh        = 0.975 ;

%DL2               = 20^2 ;

%wingLength        = 30 * params.pixPerCM / 232  % 45 / 2
wingTipVelocityThreshold = 5 ;  % voxels/frame


LL = .55 ; %0.66
LL = wingLength * LL ; % used with farthestPoint

%disp('waiting 2 second...') ;
%pause(2) ;

%enterTime         = max(params.enterTime) ;
%exitTime          = min(params.exitTime) ;
startTrackingTime = params.startTrackingTime;
endTrackingTime   = params.endTrackingTime ;
%firsts            = [params.metaData.firstImage] ;
Nimages           = endTrackingTime - startTrackingTime + 1 ;

%kmeansReplicates = 50 ;

%K = 3 ; % number of clusters in gm analysis (for kmeans K=4 was previously used)
gmReplicates = 12 ;
kmeansReplicates = 12 ;

if plotFlag
    h=figure('position',[ 424   447   900   500]) ;
    %az = -10 ;% 265 ;
    
    % calculate axis for dispaly
    axMat = zeros(3,6) ;
    axMat(1,:) = [min(bodyHull(:,2)) max(bodyHull(:,2)) min(bodyHull(:,3)) max(bodyHull(:,3)) min(bodyHull(:,4)) max(bodyHull(:,4))  ] ;
    axMat(2,:) = [min(wing1Hull(:,2)) max(wing1Hull(:,2)) min(wing1Hull(:,3)) max(wing1Hull(:,3)) min(wing1Hull(:,4)) max(wing1Hull(:,4))  ] ;
    axMat(3,:) = [min(wing2Hull(:,2)) max(wing2Hull(:,2)) min(wing2Hull(:,3)) max(wing2Hull(:,3)) min(wing2Hull(:,4)) max(wing2Hull(:,4))  ] ;
    ax        =  [min(axMat(:,1)) max(axMat(:,2)) min(axMat(:,3)) max(axMat(:,4)) min(axMat(:,5)) max(axMat(:,6)) ] ;
    clear axMat ;
    
    colmap = [0 0.8 0 ; 1 0 0 ; 0 0 1 ];
end
%------------------------------------------------------------------------
% INITIALIZE COORDINATE VARIABLES. KEEP SAME NAMES AS IN ORIGINAL VERSION
% OF THE ALGORITHM (UNLINE THAT VERSION, HERE WE PRE-ALLOCATE).
% IN THE MEANTIME, UNITS ARE PIXELS. NEED TO CONVERT TO REAL-WORLD UNITS AT
% SOME POINT (MULTIPLY VOXELSIZE)
% -----------------------------------------------------------------------

% disp('correct wing 1/2 L/R here') ;

xs = zeros(Nimages,1); % body center of mass
ys = zeros(Nimages,1);
zs = zeros(Nimages,1);

psis  = zeros(Nimages,1); % body Euler angles
betas = zeros(Nimages,1);
rhos  = zeros(Nimages,1);

x1s = zeros(Nimages,1); % wing 1 center of mass
y1s = zeros(Nimages,1);
z1s = zeros(Nimages,1);

phi1s = zeros(Nimages,1); % right wing Euler angles
theta1s = zeros(Nimages,1);
eta1s = zeros(Nimages,1);

x2s = zeros(Nimages,1); % wing 2 center of mass
y2s = zeros(Nimages,1);
z2s = zeros(Nimages,1);

phi2s   = zeros(Nimages,1); % left wing Euler angles
theta2s = zeros(Nimages,1);
eta2s   = zeros(Nimages,1);

rightWingTips = zeros(Nimages,3) ;
leftWingTips  = zeros(Nimages,3) ;

% Memory allocations for unit vectors:
% ------------------------------------
rollHats    = zeros(Nimages,3) ;

psiHats     = zeros(Nimages,3) ;
normRolls   = zeros(Nimages,3) ; %#ok<NASGU> % currently allocated just to avoid bugs in gui

span1Hats   = zeros(Nimages,3) ;
chord1Hats  = zeros(Nimages,3) ;
chord1Hat2s = zeros(Nimages,3) ;
phi1Hats    = zeros(Nimages,3) ;

span2Hats   = zeros(Nimages,3) ;
chord2Hats  = zeros(Nimages,3) ;
chord2Hat2s = zeros(Nimages,3) ;
phi2Hats    = zeros(Nimages,3) ;

% alternative chord vectors (minor diagonal of parallelogram)
chord1AltHats = zeros(Nimages,3) ;
chord2AltHats = zeros(Nimages,3) ;


diag11Right = zeros(Nimages,1);
diag12Right = zeros(Nimages,1);
diag21Left = zeros(Nimages,1);
diag22Left = zeros(Nimages,1);
% more memory allocations
rightChordTopProjections = zeros(Nimages,1) ;
leftChordTopProjections  = zeros(Nimages,1) ;


% -------------------------------------------------------------------------
% ALLOCATE A BOOLEAN MATRIX SAME SIZE AS 'RES' THAT INDICATES, FOR EACH
% VOXEL, TO WHICH PART IT BELONGS - BODY, RIGHT WING, OR LEFT WING:
%
% RESIDX(ISBODY, ISRIGHTWING, ISLEFTWING)
%
% IT IS MORE MEMORY-EFFICIENT THAN HAVING AN UNSIGNED-INT MATRIX, SINCE
% WITH BOOLEANS EACH ROW TAKES 3 BITS AND WE NEED 8 BITS WITH A SINGLE UINT
% PER ROW (1,2,3)
% -------------------------------------------------------------------------

bodyInd  = 1 ;
n1 = size(bodyHull,1) ;
n2 = size(wing1Hull,1) ;
n3 = size(wing2Hull,1) ;
s = n1 + n2 + n3  ;
%RESIDX = false(s,3) ;

RESIDX_BODY = false(n1,3) ;
RESIDX_BODY(:,bodyInd) = true ;

RESIDX_WING1 = false(n2,3) ;
RESIDX_WING2 = false(n3,3) ;

res = [bodyHull ; wing1Hull ; wing2Hull ] ; % will be sorted by time later
% in the end we will have:
% RESIDX = [ RESIDX_BODY ; RESIDX_WING1 ; RESIDX_WING2 ] ;

clear s n1 n2 n3
% -----------------------------------------------------------------
% ALLOCATE ERROR LOG, errorLog(n)=true iff image n was not analyzed
% -----------------------------------------------------------------

errorLog = false(Nimages, 1) ;

ind = tempOffset ;

prevCRight = [] ;
prevCLeft  = [] ;
prevGM     = [] ;



% define constants

%prevNclustered = -1 ;
%startTrackingTime = startTrackingTime + 1067 ;
%startTrackingTime=570 ;

for t = startTrackingTime+tempOffset : endTrackingTime %SW added +1. why??? - SW
    
    tic
    ind = ind + 1 ;
    n   = t - params.firstTrackableFrame + 1 ;
    
    if verboseFlag
        disp(' ');
        disp(['Start frame ' num2str(ind) '/' num2str(Nimages) ]) ;
    end
    i1body = frameStartIndBodyHull(n) ;
    i2body = frameEndIndBodyHull(n) ;
    bodyCoords = double(bodyHull(i1body:i2body, 2:4)) ; % (x,y,z) in voxel indices, not real coordinates
    
    i1wing1 = frameStartIndWing1Hull(n) ;
    i2wing1 = frameEndIndWing1Hull(n) ;
    wing1Coords = double(wing1Hull(i1wing1:i2wing1, 2:4)) ;
    
    i1wing2 = frameStartIndWing2Hull(n) ;
    i2wing2 = frameEndIndWing2Hull(n) ;
    wing2Coords = double(wing2Hull(i1wing2:i2wing2, 2:4)) ;
    
    % CHECK IF WINGS ARE EMPTY
    wing1empty = isnan(wing1Coords(1,1)) ;
    wing2empty = isnan(wing2Coords(1,1)) ;
    
    
    % if the two wings hull are the same, use clustering to distinguish
    % both wings
    if (mergedWingsFlag(n) && ~wing1empty && ~wing2empty)
        if verboseFlag
        disp('Two wings are merged in the XY view. Cluster using kmeans.') ;
        end
        %keyboard ;
        % the two wing hulls should be the same. take the first.
        
        wing12Coords = double(wing1Coords) ;
        
        %if (oneWingEmptyFlags(n))
        % remove body-hull voxels from wings hull
        keepRows = ~ismember(wing12Coords, bodyCoords,'rows') ;
        wing12Coords = wing12Coords(keepRows,:) ;
        clear keepRows ;
        %end
        
        
        if (ind==1) % only for first frame, do not use seed
            IDX = kmeans(wing12Coords, 2,'replicates', kmeansReplicates) ;
        else
            % calculate initial guess for the two cenroids
            seedMat =  [ x1s(ind-1) y1s(ind-1) z1s(ind-1) ; ...
                x2s(ind-1) y2s(ind-1) z2s(ind-1) ] ;
            
            try
                IDX = kmeans(wing12Coords, 2,'start', seedMat) ;
            catch %#ok<CTCH>
                if verboseFlag
                    disp('kmeans clustering based on previous centroid failed') ;
                    disp('try to find the closest points to the previous centeroids')
                    disp('and use them as the initial guess');
                end
                Q = size(wing12Coords,1) ;
                dst = myNorm ( wing12Coords - repmat(seedMat(1,:),Q,1)) ;
                [~, q1] = min(dst) ;
                dst = myNorm ( wing12Coords - repmat(seedMat(2,:),Q,1)) ;
                [~, q2] = min(dst) ;
                seedMat = wing12Coords([q1 q2],:) ;
                try
                    IDX = kmeans(wing12Coords, 2,'start', seedMat) ;
                catch %#ok<CTCH>
                    % if, for some reason, the seed matrix did not work,
                    % just cluster using the default method
                    if verboseFlag
                        disp('Seed matrix did not work. Use random-search.') ;
                    end
                    IDX = kmeans(wing12Coords, 2,'replicates', kmeansReplicates) ;
                end
                clear Q dst q1 q2
            end
            
            clear seedMat ;
        end
        wing1Coords = wing12Coords(IDX==1,:) ;
        wing2Coords = wing12Coords(IDX==2,:) ;
    end
    
    
    %{
    % OLD CODE USING GMDISTRIBUTION. THE ADVANTAGE OF KMEANS IS THAT IT
    % TENDS TO GIVE CLUSTERS OF SIMILAR SIZES, WHICH IS GOOD WHEN WE HAVE
    % ONLY TWO WINGS AND NOT BODY.
    if (mergedWingsFlag(n))
        %keyboard ;
        
        disp('Two wings are merged in the XY view. Try to cluster.') ;
        % the two wing hulls should be the same. take the first.
        wing12Coords = wing1Coords ;
        
        if (prevNclustered==n-1)
            %%if the previous frame was clustered
            disp('cluster based on previous frame') ;
            gm  = gmdistribution.fit(wing12Coords,2,'replicates', 1,'start',prevGM) ;
        else
            gm  = gmdistribution.fit(wing12Coords,2,'replicates', gmReplicates) ; % fit to TWO clusters
        end
        %post = posterior(gm,wing1Coords) ;
        IDX = cluster(gm,single(wing12Coords));
        wing1Coords = wing12Coords(IDX==1,:) ;
        wing2Coords = wing12Coords(IDX==2,:) ;
        
        prevNclustered     = n ;
        prevGM.PComponents = gm.PComponents ;
        prevGM.Sigma       = gm.Sigma ;
        prevGM.mu          = gm.mu ;
    end
    %}
    
    
    
    
    %{
    
    % note that now K=3 not 4
    if (isempty(prevGM))
        gm  = gmdistribution.fit(coords,K,'replicates', gmReplicates) ; % fit to three clusters
    else
        gm  = gmdistribution.fit(coords,K,'replicates', 1,'start',prevGM) ; % fit to three clusters
    end
    
    prevGM.PComponents = gm.PComponents ;
    prevGM.Sigma       = gm.Sigma ;
    prevGM.mu          = gm.mu ;
    
    % note that we later need to trace which cluster of the ORIGNAL ones is
    % body, rightWing and leftWing, to be able to use the probabilities
    % matrix 'post':
    post = posterior(gm,coords) ;
    
    % post(:, postCluterInd(k)) are the probabilities of belonging to
    % cluster k for all voxels
    postClusterInd = zeros(1,3) ;
    
    IDX = cluster(gm,single(coords));
            
    clusterSize = zeros(1,K) ;
    C = zeros(K,3) ; % (x,y,z)
    % calc the number of voxels in each cluster and centroid positions
    for j=1:K
        currRows = (IDX==j) ;
        clusterSize(j) = sum(currRows) ;
        C(j,:) = mean(coords(currRows,:)) ;
    end
    clear currRows ;
    %}
    
    %{
    [~, I] = sort(clusterSize,'descend') ;
    newIDX = zeros(size(IDX),'uint8');
    newCentroids = zeros(K,3) ; % (x,y,z)
    %}
    
    %{
    bodyRows = (IDX==I(1)) ; % largest cluster only
    newIDX(bodyRows) = bodyInd ;
    postClusterInd(bodyInd) = I(1) ;
    
    newCentroids(bodyInd,:) = C(I(1),:) ;
    
    
    wing1Rows = (IDX==I(2)) ; % second in size (wing)
    newIDX(wing1Rows) = wing1Ind ;
    newCentroids(wing1Ind,:) = C(I(2),:) ;
    postClusterInd(wing1Ind) = I(2) ;
    
    
    wing2Rows = (IDX==I(3)) ; % smallest (other wing)
    newIDX(wing2Rows) = wing2Ind ;
    newCentroids(wing2Ind,:) = C(I(3),:) ;
    postClusterInd(wing2Ind) = I(3) ;
    %}
    
    bodyInd  = 1 ;
    wing1Ind = 2 ; % we will later decide which wing is left/right
    wing2Ind = 3 ;
    
    rightWingInd  = 2 ; % will be used later on when L/R is determined
    leftWingInd   = 3 ;
    
    newCentroids = zeros(3,3) ; % (x,y,z) for 3 clusters (rows)
    
    
    % calculate centroids
    newCentroids(bodyInd,:) = mean(bodyCoords) ;
    
    % calculation of the wings centroids is no longer based on simple
    % center of mass. see below.
    %newCentroids(wing1Ind,:) = mean(wing1Coords) ;
    %newCentroids(wing2Ind,:) = mean(wing2Coords) ;
    
    cBody = newCentroids(bodyInd,:) ;
    
    % calculate wing centroids based on "tip"
    % ---------------------------------------
    
    % use the name "farPoint" since it is different than the "tip" we will
    % find later on.
    
    % first find the largest connected component of the wing voxels
    % (first use of this function)
    if (~wing1empty)
        wing1LargestCC_initial = findLargestHullCC (wing1Coords);
        
        % from the largest CC, find the farpoint and then select the voxels
        % that are close to the farpoint.
        [farPoint1, ~, list1] = farthestPoint(wing1LargestCC_initial, cBody, LL) ;
        
        
        % find the largest connected component within the voxels in list1.
        % list1 are the indices of the voxles in wing1LargestCC_initial whose distance
        % from "farPoint" is smaller than LL.
        %
        % largest component is calculated AGAIN, to account for a case where cutting
        % the LL distance left us with two connected components
        wing1LargestCC = findLargestHullCC (wing1LargestCC_initial(list1,:));
        
        clear wing1LargestCC_initial
        
        % make sure the far point is included in the connected components
        % if not, take another one
        if (~ismember(farPoint1, wing1LargestCC,'rows'))
            %keyboard
            if verboseFlag
                disp('problem 1?') ;
            end
            tmp = setdiff( wing1Coords(list1,:), wing1LargestCC, 'rows') ;
            wing1LargestCC = findLargestHullCC(tmp) ;
            clear tmp
        end
        
        % check again
        if (~ismember(farPoint1, wing1LargestCC,'rows')) && verboseFlag
            disp('problem 2?');
            %keyboard ;
        end
        
        %newCentroids(wing1Ind,:) = mean(wing1LargestCC) ;
        
        newCentroids(wing1Ind,:) = ...
            calcCentroidFromTopView(wing1LargestCC, farPoint1, wingLength/3) ;
        
    end
    
    % WING 2
    % ------
    if (~wing2empty)
        % first find the largest connected component of the wing voxels
        % (first use of this function)
        wing2LargestCC_initial = findLargestHullCC (wing2Coords);
        
        % from the largest CC, find the farpoint and then select the voxels
        % that are close to the farpoint.
        [farPoint2, ~, list2] = farthestPoint(wing2LargestCC_initial, cBody, LL) ;
        
        % find the largest connected component within the voxels in list2.
        % list2 are the indices of the voxles in wing2LargestCC_initial whose distance
        % from "farPoint" is smaller than LL.
        %
        % largest component is calculated AGAIN, to account for a case where cutting
        % the LL distance left us with two connected components
        wing2LargestCC = findLargestHullCC (wing2LargestCC_initial(list2,:));
        
        clear wing2LargestCC_initial
        % make sure the far point is included in the connected components
        % if not, take another one
        if (~ismember(farPoint2, wing2LargestCC,'rows'))
            %keyboard
            tmp = setdiff( wing2Coords(list2,:), wing2LargestCC, 'rows') ;
            wing2LargestCC = findLargestHullCC(tmp) ;
            clear tmp
        end
        
        % check again
        if (~ismember(farPoint2, wing2LargestCC,'rows')) && verboseFlag
            disp('problem 3?');
            %keyboard ;
        end
        
        
        %newCentroids(wing2Ind,:) = mean(wing2LargestCC) ;
        newCentroids(wing2Ind,:) = ...
            calcCentroidFromTopView(wing2LargestCC, farPoint2, wingLength/3) ;
        
        % combine three hulls into "coords" and specify bodyRows, wing1Rows and
        % wing2Rows
        n1 = size(bodyCoords,1) ;
        n2 = size(wing1Coords,1) ;
        n3 = size(wing2Coords,1) ;
        
        % note there might be a duplicity between "coords", "wingXLargestCC"
        coords = [ bodyCoords ; wing1Coords ; wing2Coords ] ;
        nn = n1+n2+n3 ;
        % use logical indexing (should be faster than numeric)
        bodyRows  = false(nn,1) ; bodyRows(1:n1)                  = true ; % bodyRows  = 1:n1 ;
        wing1Rows = false(nn,1) ; wing1Rows((n1+1):(n1+n2))       = true ;
        wing2Rows = false(nn,1) ; wing2Rows((n1+n2+1):(n1+n2+n3)) = true ;
        
        clear n1 n2 n3 nn
    end
    % ----------------------------------------------
    % TRACKING BODY (LINES 653-673 IN ORIGINAL FILE)
    % ----------------------------------------------
    
    xs(ind)   = newCentroids(bodyInd,1) ;
    ys(ind)   = newCentroids(bodyInd,2) ;
    zs(ind)   = newCentroids(bodyInd,3) ;
    
    pcaCoeffs = pca(coords(bodyRows,:));     % PCA on body voxels
    
    % 1st axis = main body axis (roll axis, e.g. Ahat)
    % (the other principal components of the body are unreliable)
    try
        rollHat   = pcaCoeffs(:,1);
    catch
        continue
    end
    
    % make roll axis have positive z-component (meaning it points upward)
    % NOTE THAT IN EXTREME MANEUVERS THIS MIGHT BE WRONG
    %{
    tempRhoProj = [rollHat(1), rollHat(2), 0] ;
    tempRhoProj = tempRhoProj / norm(tempRhoProj) ; %hat vector
    delInd = 5 ; %number of time steps to look back to define velocity, 5 picked arbitrarily
    if ind > delInd
        velocityProj = [xs(ind) - xs(ind - delInd), ys(ind) - ys(ind - delInd), 0] ;
        velocityProj = velocityProj / norm(velocityProj) ;
    else
        velocityProj = double([bodyHull(end,2) - bodyHull(1,2), bodyHull(end,3) - bodyHull(1,3), 0]) ;
        velocityProj = velocityProj / norm(velocityProj) ;
    end
    
    if (dot(tempRhoProj,velocityProj)<0) %(rollHat(3)<0) &&
        rollHat = -rollHat ;
    end
    %}
    
    % check sign of body axis
    if ((t <= 0)  && (rollHat(3)<0)) || ((ind <= 1) && (rollHat(3)<0))
        rollHat = -rollHat ;
    elseif (t > 0) && (ind > 1)
        if (dot(rollHat,rollHats(ind-1,:)') <= 0)
            rollHat = -rollHat ;
        end
    end
    
    psi  = atan2(rollHat(2),rollHat(1)); % body angle with respect to x axis
    beta = asin(rollHat(3));             % body angle with respect to the horizon
    
    % check if these are used later on
    rhoProj = [rollHat(1),rollHat(2),0]; % projection of Ahat onto xy plane
    rhoProj = rhoProj / norm(rhoProj);   % normalize it to be a unit vector
    psiHat  = [-sin(psi),cos(psi),0];
    betaHat = -rhoProj*sin(beta) + [0,0,1]*cos(beta);
    
    psis(ind)  = psi;
    betas(ind) = beta;
    
    % SKIP FINDING THE BODY ROLL ANGLE, AS DEFINED IN THE ORIGINAL CODE.
    % THIS ANGLE IS UNRELIABLE ANYWAY.
    % IF USING A FLY+PIN THIS ANGLE COULD BE EXTRACTED BY FINDING THE
    % ORIENTATION OF THE PIN
    
    % -----------------------------------------------------------------
    % IDENTIFY WHICH WHICH IS LEFT AND WHICH IS RIGHT
    % from here on we use leftWingInd (2) and rightWingInd (3), and the
    % correspoinding leftWingRows and rightWingRows
    % -----------------------------------------------------------------
    if (~wing1empty && ~wing2empty)
        % 1 and 2 indices for the wings are not yet assigned. here we
        % determine if we need to swap or not.
        
        % centroids for both wings
        c1 = newCentroids(wing1Ind,:);
        c2 = newCentroids(wing2Ind,:);
        cBody = newCentroids(bodyInd,:) ;
        
        
        % alternative swap code
        v1 = c1 - cBody ; v1 = v1 / norm(v1) ;
        v2 = c2 - cBody ; v2 = v2 / norm(v2) ;
        
        %val1 = dot ( cross(v1, rhoProj), rollHat) ;
        %val2 = dot ( cross(v2, rhoProj), rollHat) ;
        val1 = dot ( cross(v1, rhoProj), [0 0 1]) ;
        val2 = dot ( cross(v2, rhoProj), [0 0 1]) ;
        alternativeSwapFlag = false ;
        
        if (val1>0 && val2<0) % both have the "correct" signs
            alternativeSwapFlag = false ;
        elseif ( val1<0 && val2>0 ) % both have the incorrect signs
            alternativeSwapFlag = true ;
        else % both have the same sign
            % just verify
            if (val1*val2<0) && verboseFlag
                disp('problem 4? - both should have the same sign') ;
                %keyboard ;
            end
            dot1 = dot(v1, rhoProj) ;
            dot2 = dot(v2, rhoProj) ;
            
            if (val1>0 && val2>0) % both are positive
                % check if we need to swap - if dot1 is right, then it should
                % be closer to rhoProj than dot2
                if (dot1<dot2)
                    alternativeSwapFlag = true ;
                end
            else % both are negative
                % check if we need to swap if dot1 is right it should be
                % farther to rhoProj than dot2
                if (dot2<dot1)
                    alternativeSwapFlag = true ;
                end
            end
        end
        
        %clear v1 v2 val1 val2 dot1 dot2
        
        if (alternativeSwapFlag)
            if verboseFlag
                disp('===> Swap wings based on vector calculation');
            end
            % we need to swap the wings
            rightWingRows = wing2Rows ;
            leftWingRows  = wing1Rows ;
            %newIDX(leftWingRows) = leftWingInd ;
            %newIDX(rightWingRows) = rightWingInd ;
            newCentroids = [cBody ; c2 ; c1] ;
            cRight = c2 ;
            cLeft  = c1 ;
            %listRight = list2 ;
            %listLeft  = list1 ;
            farPointRight = farPoint2 ;
            farPointLeft  = farPoint1 ;
            rightWingLargestCC = wing2LargestCC ;
            leftWingLargestCC  = wing1LargestCC ;
            
            RESIDX_WING1(i1wing1:i2wing1, leftWingInd)  = true ;
            RESIDX_WING2(i1wing2:i2wing2, rightWingInd) = true ;
        else
            % do not swap wings. only need to define the rows (logical) indices
            rightWingRows = wing1Rows ;
            leftWingRows  = wing2Rows ;
            cRight = c1 ;
            cLeft  = c2 ;
            %listRight = list1 ;
            %listLeft  = list2 ;
            farPointRight = farPoint1 ;
            farPointLeft  = farPoint2 ;
            rightWingLargestCC = wing1LargestCC ;
            leftWingLargestCC  = wing2LargestCC ;
            
            RESIDX_WING1(i1wing1:i2wing1, rightWingInd) = true ;
            RESIDX_WING2(i1wing2:i2wing2, leftWingInd)  = true ;
        end
        
        
        % OLD WING-SWAP CODE
        %{
    % rollHat  is Ahat (body axis)
    if (dot( cross( c1'-cBody' , rollHat) , [0,0,1]) < 0)
        % we need to swap the wings
        rightWingRows = wing2Rows ;
        leftWingRows  = wing1Rows ;
        %newIDX(leftWingRows) = leftWingInd ;
        %newIDX(rightWingRows) = rightWingInd ;
        newCentroids = [cBody ; c2 ; c1] ;
        cRight = c2 ;
        cLeft  = c1 ;
        % swap also postClusterInd
        %tmp = postClusterInd(wing1Ind) ;
        %postClusterInd(wing1Ind) = postClusterInd(wing2Ind) ;
        %postClusterInd(wing2Ind) = tmp ;
        %clear tmp ;
        disp('Swap 1') ;
    else
        % do not swap wings. only need to define the rows (logical) indices
        rightWingRows = wing1Rows ;
        leftWingRows  = wing2Rows ;
        cRight = c1 ;
        cLeft  = c2 ;
    end
    
        %}
        
        %if (t==-838 || t==-837 || t==-872)
        %    keyboard ;
        %end
        
        % verify swapping if the previous left/right centroids are defined
        if (false) % (~isempty(prevCRight))
            % calc distances squared
            d11 = sum( (cRight - prevCRight).^2) ;
            d12 = sum( (cRight - prevCLeft) .^2) ;
            d21 = sum( (cLeft  - prevCRight).^2) ;
            d22 = sum( (cLeft  - prevCLeft).^2 ) ;
            % current right wing should be closer to the previous right wing
            % that to the previous left wing
            if (d11>d12 && d22>d21)
                % we need to swap again
                % swap centoids
                if verboseFlag
                    disp('Warning: Wing distances do not make sense. check this.')
                    disp('also check if other variables need to be swapped');
                    disp('problem 5?') ;
                end
                %keyboard ;
                cTemp  = cRight ;
                cRight = cLeft ;
                cLeft  = cTemp ;
                rowsTemp      = rightWingRows ;
                rightWingRows = leftWingRows ;
                leftWingRows  = rowsTemp ;
                %newIDX(leftWingRows)  = leftWingInd ;
                %newIDX(rightWingRows) = rightWingInd ;
                newCentroids = [cBody ; cRight ; cLeft] ;
                
            end
            
            %{
        % check if wings have moved too much between the previous frame and
        % the current one. If so, reset the prevGM to force a re-clustering
        % and also reset prevCRight and prevCLeft to avoid entering into
        % this part of the code in the next loop
        %
        % NOTE - THIS MECHANISM WAS CHANGE, SINCE WE DO NOT USE CLUSTERING
        % AS THE MAIN METHOD FOR SEGMENTING THE WINGS. RATHER, WE RELY ON
        % THE SEGMENTATION DONE IN THE XY VIEW.
       
        % recalc distances in case we switched wings
        d11 = sum( (cRight - prevCRight).^2) ;
        d22 = sum( (cLeft  - prevCLeft).^2 ) ;
        if (d11>DL2 || d22>DL2)
            prevGM = [] ;
            prevCRight = cRight ; % [] ;
            prevCLeft =  cLeft ; % [] ;
        end
            %}
        end
        
        % throw away the wing voxels that were identified in probability
        % smaller than probThresh
        %{
    if (0)
        rightWingRows = rightWingRows & (post(:, postClusterInd(rightWingInd)) > probThresh) ;
        leftWingRows  = leftWingRows  & (post(:, postClusterInd(leftWingInd)) > probThresh) ;
        % RECALC WING CENTROIDS
        newCentroids(rightWingInd,:) = mean(coords(rightWingRows,:)) ;
        newCentroids(leftWingInd, :) = mean(coords(leftWingRows,: )) ;
        cRight = newCentroids(rightWingInd,:) ;
        cLeft  = newCentroids(leftWingInd, :) ;
    end
        %}
        
        %{
    % DON'T USE THE FOLLOWING PART
    %
    % verify that the centroid of each section (body, right wing, left wing)
    % is located inside its section. if not, issue a warning
    
    % body
    roundedCentroid = round(newCentroids(bodyInd,:)) ;
    foundBody = ~isempty ( find (coords(bodyRows,1)==roundedCentroid(1) & ...
        coords(bodyRows,2)==roundedCentroid(2) & ...
        coords(bodyRows,3)==roundedCentroid(3) ) ) ;  %#ok<EFIND>
    if (~foundBody)
        disp(['----> t=' num2str(t) ' n=' num2str(n) ' Bad clustering warning (body)']) ;
    end
    
    % right wing
    roundedCentroid = round(newCentroids(rightWingInd,:)) ;
    foundRightWing = ~isempty ( find( coords(rightWingRows,1)==roundedCentroid(1) & ...
        coords(rightWingRows,2)==roundedCentroid(2) & ...
        coords(rightWingRows,3)==roundedCentroid(3) ) ) ;  %#ok<EFIND>
    if (~foundRightWing)
        disp(['----> t=' num2str(t) ' n=' num2str(n) ' Bad clustering warning (right wing)']) ;
    end
    
    % left wing
    roundedCentroid = round(newCentroids(leftWingInd,:)) ;
    foundLeftWing = ~isempty ( find( coords(leftWingRows,1)==roundedCentroid(1) & ...
        coords(leftWingRows,2)==roundedCentroid(2) & ...
        coords(leftWingRows,3)==roundedCentroid(3) ) ) ;  %#ok<EFIND>
    if (~foundLeftWing)
        disp(['----> t=' num2str(t) ' n=' num2str(n) ' Bad clustering warning (left wing)']) ;
    end
    
    if (foundBody && foundRightWing && foundLeftWing)
        prevCRight = cRight ;
        prevCLeft  = cLeft  ;
    end
        %}
        
        prevCRight = cRight ;
        prevCLeft  = cLeft  ;
        
        
        %{
    % check if right wing consists of more than one connected component. if
    % so, define the large component as the wing and append the smaller one
    % to the body
        
    disp('Start clustering right wing voxels...') ;
    [ ~, clustersRight ] = buildConnGraph_3D(coords(rightWingRows,:), 1) ;
    disp('Done clustering right wing voxels.') ;
        %}
        
        % CURRENT CODE USES RESIDS_BODY, RESIDX_WING1, RESIDX_WING2
        % WHILE RESIDX IS CONSTRUCTED IN THE END.
        
        % make sure the following are not used later on and cause a mess
        clear wing1Ind wing1Rows wing2Ind wing2Rows c1 c2
        
        %{
    % STORE VOXEL CLUSTER CLASSIFICATION: BODY/LEFTWING/RIGHTWING
    %
    RESIDX(i1:i2,:) = [ bodyRows rightWingRows leftWingRows] ;

    % Now, the right wing voxels are in:  coords(rightWingRows)
    %           left wing voxels are in:  coords(leftWingRows)
    % the rows should satisfy
    % rightWingRows = (newIDX==rightWingInd)
    % leftWingRows  = (newIDS==leftWingInd) ;
        %}
        
        % -----------
        % TRACK WINGS
        % -----------
        
        % -------------------------
        % START WITH THE RIGHT WING
        % -------------------------
        
        
        % pick a subset of the wing voxels closer to the tip. Namely, use
        % rightWingLargestCC which we calculated before
        rightWingVoxels = coords(rightWingRows,:) ;
        
        %{
    % PCA ON RIGHT WING VOXELS
    pcaCoeffs      = princomp(double(rightWingLargestCC));  % prev version
    spanHatRight   = pcaCoeffs(:,1); % 1st axis
    secondHatRight = pcaCoeffs(:,2); %#ok<NASGU> % 2nd axis
        %}
        
        % FIRST ESTIMATION OF SPAN VECTOR USING FARTHEST POINT
        % THIS ESTIMATION WILL BE UPDATED BELOW USING THE NEW WING TIP CALC
        
        % use a new "definition" of the span vector, as the unit vector from
        % the wing's center of mass to its farthest point (then change the
        % latter to be the wing tip).
        farPointR    = farthestPoint(rightWingLargestCC, cBody, LL) ;
        spanHatRight = farPointR - cRight ;
        spanHatRight = spanHatRight' ;
        spanHatRight = spanHatRight / norm(spanHatRight) ;
        
        % makes wing span point outward
        if dot(cRight-cBody,spanHatRight) < 0
            spanHatRight = -spanHatRight;
        end
        
        
        
        % FIND WING TIP
        % -------------
        
        rightWingTip = findWingTip(rightWingLargestCC, spanHatRight', cRight);
        if (isnan(rightWingTip(1)))
            rightWingTip = farPointR ;
        end
        
        % recalculate span vector based on the refined wing tip
        spanHatRight = rightWingTip - cRight ;
        spanHatRight = spanHatRight' ;
        spanHatRight = spanHatRight / norm(spanHatRight) ;
        
        span1Hat     = spanHatRight ; % get rid of this duplicity
        
        
        %{
    % OLD: RECALC WING CENTER OF MASS AS WINGLENGTH/2 AWAY FROM THE TIP
    cRight = rightWingTip - span1Hat'*wingLength/2 ;
    newCentroids(rightWingInd,:) = cRight ;
        %}
        
        % store right wing center of mass (centroid)
        x1s(ind) = cRight(1) ; % newCentroids(rightWingInd,1) ;
        y1s(ind) = cRight(2) ; % newCentroids(rightWingInd,2) ;
        z1s(ind) = cRight(3) ; % newCentroids(rightWingInd,3) ;
        
        %{
    % SEE IF THIS PIECE OF CODE IS RELEVANT. IT IS RELATED TO THE CODE IN
    % LINES 594-650 IN THE ORIGINAL FILE
    
    % if wings are obscured at back just take the span to be the vector in
    % the plane of the wing that connects body center to wing center
    if iterClust == 1
        third1Hat = cross(span1Hat,second1Hat);
        third1Hat = third1Hat/norm(third1Hat);
        span1Hat = c1'-cBody' - dot(c1'-cBody',third1Hat)*third1Hat;
        span1Hat = span1Hat/norm(span1Hat);
    end
        %}
        
        % right wing phi and theta.
        % NOTE: here phi is with respect to the x axis in LAB FRAME OF REF.
        phi1   = atan2(spanHatRight(2),spanHatRight(1));
        theta1 = asin(spanHatRight(3));
        
        phi1Proj = [span1Hat(1),span1Hat(2),0];
        phi1Proj = phi1Proj / norm(phi1Proj);
        phi1Hat = [-sin(phi1),cos(phi1),0];
        theta1Hat = -phi1Proj*sin(theta1) + [0,0,1]*cos(theta1);
        
        % right wing eta (pitch angle)
        % ----------------------------
        
        %rightWingVoxels = coords(rightWingRows,:) ;
        
        try
            if verboseFlag
                disp('Right wing:') ;
            end
            rightWingVoxels = double(rightWingLargestCC) ;
            Nvox            = size(rightWingVoxels,1) ;
            
            mat1 = rightWingVoxels - repmat(cRight, Nvox,1) ;
            mat2 = repmat(spanHatRight', Nvox, 1) ;
            
            distFromMidSpan = abs(sum(mat1.*mat2,2) ) ;
            clear mat1 mat2
            
            chordRowsInd = find(distFromMidSpan<delta) ;
            
            if (isempty(chordRowsInd))
                % first try a larger delta
                chordRowsInd = find(distFromMidSpan<3*delta) ;
                % check if still empty
                if (isempty(chordRowsInd))
                    error('hullAnalysis:Chord','Bad clustering - empty right chord') ;
                end
            end
            clear distFromMidSpan
            % among the chordRowsInd voxels, find the pair that is the most distant
            % apart. this pair defines the direction of the chord
            % original code calculated a distance matrix between all pairs. here we
            % do it a bit more efficiently by excluding most of the voxels before
            % calculating the distance matrix
            
            % caculate the distance^2 of each voxel from the wing centroid
            Nvox    = length(chordRowsInd);
            mat1    = (rightWingVoxels(chordRowsInd,:) - repmat(cRight, Nvox,1)).^2 ;
            distVec = (sum(mat1,2)).^0.5 ;
            clear mat1
            
            % select only the top quarter of the voxels, i.e the most distant from
            % wing centroid
            
            [~, sortedInd] = sort(distVec,'descend') ;
            selectedInd    = chordRowsInd(sortedInd(1:ceil(Nvox*chordFraction))) ;
            %selectedIndRight = selectedInd ;
            clear distVec
            
            % find the most distant pair
            
            distMat = squareform (pdist (rightWingVoxels(selectedInd,:))) ;
            
            [maxRowVec Irow] = max(distMat,[],1) ;
            [~, Icol] = max(maxRowVec) ;
            Irow = Irow(Icol) ;
            
            if (distMat(Irow, Icol)~=max(distMat(:))) && verboseFlag
                disp('Error with finding max. plz check.') ;
                disp('problem 6?') ;
                %keyboard ;
            end
            
            
            % the following indices give voxel coordinate:
            vox1IndRight = selectedInd(Irow) ; % voxel position is rightWingVoxels(vox1IndRight,:)
            vox2IndRight = selectedInd(Icol) ; % voxel position is rightWingVoxels(vox2IndRight,:)
            
            chord1Hat = rightWingVoxels(vox1IndRight,:)' - rightWingVoxels(vox2IndRight,:)' ;
            
            % force chord to be vertical to the span vector
            chord1Hat = chord1Hat - span1Hat * dot(span1Hat, chord1Hat) ;
            
            diag11 = norm(chord1Hat) ; % will be used later
            
            chord1Hat = chord1Hat / norm(chord1Hat) ;
            
            if (chord1Hat(3)<0)
                chord1Hat = - chord1Hat ;
            end
            
            
            % FIND THE SECOND DIAGONAL OF THE RIGHT WING PARALLELOGRAM (ADDED FEB 2012)
            % ------------------------------------------------------------------------
            
            % see http://mathworld.wolfram.com/Point-PlaneDistance.html
            mat1    = rightWingVoxels(chordRowsInd,:) - repmat(cRight, Nvox,1) ;
            % mat1 contains the vectors connecting the voxels
            % in chordRowsInd to the wing centoid.
            
            % calc the vector normal to the span and chord
            wingNormVec = cross(span1Hat, chord1Hat) ;
            
            % calc the (signed) distance from each point in mat1 to the wing plane
            mat2 = repmat(wingNormVec', Nvox, 1) ;
            distVec = sum(mat1.*mat2,2) ;
            
            % find the largest positive and largest negative distances, which
            % correspond to the farthest voxels on each side of the plane
            
            [maxval, indmax] = max(distVec) ; % index into rightWingVoxels
            [minval, indmin] = min(distVec) ; % index into rightWingVoxels
            
            if (maxval<=0 || minval>=0) && verboseFlag
                disp('error in finding alternative chord vector for right wing') ;
                %keyboard ;
            end
            
            indmin = chordRowsInd(indmin) ;
            indmax = chordRowsInd(indmax) ;
            chord1AltHat = rightWingVoxels(indmax,:)' - rightWingVoxels(indmin,:)' ;
            
            % force alternative chord to be vertical to the span vector
            chord1AltHat = chord1AltHat - span1Hat * dot(span1Hat, chord1AltHat) ;
            
            diag12 = norm(chord1AltHat) ; % will be used later
            % now normalize
            chord1AltHat = chord1AltHat / norm(chord1AltHat) ;
            
            % choose the sign of the alternative chord vector such that it is
            % has positive overlap with the "main" chord vector
            
            %if ( dot(chord1AltHat, chord1Hat) < 0 )
            %    chord1AltHat = - chord1AltHat ;
            %end
            
            if (chord1AltHat(3)<0)
                chord1AltHat = - chord1AltHat ;
            end
            
            
            diag11Right(ind) = diag11 ;
            diag12Right(ind) = diag12 ;
            
            
            % decide whether to swap the "main" and "alternative" chord vectors
            % -----------------------------------------------------------------
            if verboseFlag
                disp(['Before swap: diag11=' num2str(diag11) ' diag12=' num2str(diag12)]) ;
                disp(['diag11/diag12=' num2str(diag11/diag12) '   diag12/diag11=' num2str(diag12/diag11)]) ;
            end
            % if one of the diagonals is siginficantly longer, choose the longer
            % one and do not proceed to the velocity criterion below
            diagSwapFlag     = false ;
            velocitySwapFlag = false ;
            
            if (diag12/diag11 >= 1.3)
                diagSwapFlag = true ;
                %contProcess = false ;
                if verboseFlag
                    disp('swap based on large ratio') ;
                end
            end
            
            % find wingtip 'velocity' with respect to the body
            if(ind>1)
                % previous version calculate wing centroid velocity:
                vWingCM = [  (x1s(ind) - xs(ind)) - (x1s(ind-1)-xs(ind-1)) ; ...
                    (y1s(ind) - ys(ind)) - (y1s(ind-1)-ys(ind-1))  ; ...
                    (z1s(ind) - zs(ind)) - (z1s(ind-1)-zs(ind-1)) ] ;
                
                vWing =  ( rightWingTip - [xs(ind) ys(ind) zs(ind)] ) - ...
                    ( rightWingTips(ind-1,:) - [xs(ind-1) ys(ind-1) zs(ind-1)]) ;
                
                % keep only the component perpendicular to the span vector
                vWing = vWing - span1Hat' * dot(span1Hat, vWing) ;
                
                
                nrm = norm(vWing) ;
                if verboseFlag
                    disp(['Wing tip velocity = ' num2str(nrm)]) ;
                    disp(['Wing CM  velocity = ' num2str(norm(vWingCM))]) ;
                end
            else
                nrm = 0 ;
            end
            
            if (ind>1)
                if (nrm~=0)
                    vWing = vWing / nrm ;
                    dot1 = dot(chord1Hat, vWing) ;
                    dot2 = dot(chord1AltHat, vWing) ;
                    if verboseFlag
                        disp(['dot1=' num2str(dot1) '  dot2=' num2str(dot2)]) ;
                    end
                    if (dot2>dot1) % swap
                        velocitySwapFlag = true ;
                    end
                    if (dot1<0 && dot2<0 && nrm>=wingTipVelocityThreshold && ~velocitySwapFlag)
                        if verboseFlag
                            disp('--> inverting right chord. not swapping.') ;
                        end
                        chord1Hat = - chord1Hat ;
                    end
                end
            end
            
            swapFlag = (velocitySwapFlag && nrm>=wingTipVelocityThreshold) || ... % believe velocity if |v|>2
                (diagSwapFlag && nrm<wingTipVelocityThreshold) ;
            
            if (velocitySwapFlag && nrm>=wingTipVelocityThreshold) && verboseFlag
                disp('Right wing chord - velocity swap') ;
            end
            if (diagSwapFlag && nrm<wingTipVelocityThreshold) && verboseFlag
                disp('Right wing chord - diag swap') ;
            end
            
            % probably need a smarter way to "weigh" the two types of swaps
            % see, e.g. frame 14 where the wings should have been swapped
            if (swapFlag)
                tmp = chord1Hat ;
                chord1Hat = chord1AltHat ;
                chord1AltHat = tmp ;
                
                tmp = diag11Right(ind) ;
                diag11Right(ind) = diag12Right(ind) ;
                diag12Right(ind) = tmp ;
                
                clear tmp ;
                if verboseFlag
                    disp('Swapped right chord with alternative chord') ;
                end
            end
            
            % ----------
            
            
            % makes chord look good on right wing during downstroke
            %chord1Hat = abs(dot(chord1Hat,phi1Hat))*phi1Hat + abs(dot(chord1Hat,theta1Hat))*theta1Hat;
            
            chord1Hat2 = -abs(dot(chord1Hat,phi1Hat))*phi1Hat + abs(dot(chord1Hat,theta1Hat))*theta1Hat;
            
            % in the first quadrant only
            eta1 = atan2( abs(dot(chord1Hat,theta1Hat)), abs(dot(chord1Hat,phi1Hat)));
            
            
        catch %#ok<CTCH>
            if verboseFlag
                disp('error occured while trying to find right chord. taking values from the previous frame.')
            end
            % this ignores the case where the error occures on the first frame...
            vox1IndRight = [] ;
            vox2IndRight = [] ;
        end
        
        
        
        [p1R p2R rightChordTopProj] = chordTopViewProjection(spanHatRight, cRight, rightWingLargestCC) ;
        
        
        
        
        % -------------------------
        % NOW WORK ON THE LEFT WING
        % -------------------------
        
        %try
        % pick a subset of the wing voxels closer to the tip
        leftWingVoxels = coords(leftWingRows,:) ;
        
        %[~,~, list] = farthestPoint(leftWingVoxels, cBody, LL) ;
        
        %%1  [leftWingLargestCC, ~] = findLargestHullCC (leftWingVoxels(listLeft,:));
        
        % refine wing center of mass
        % cLeft = mean(leftWingVoxels(list,:)) ; % already done above
        
        %{
    % PCA ON LEFT WING VOXELS
    pcaCoeffs      = princomp(double(leftWingLargestCC));
    spanHatLeft   = pcaCoeffs(:,1); % 1st axis
    secondHatLeft = pcaCoeffs(:,2); %#ok<NASGU> % 2nd axis
        %}
        
        % use a new "definition" of the span vector, as the unit vector from
        % the wing's center of mass to its farthest point.
        farPointL = farthestPoint(leftWingLargestCC, cBody, LL) ;
        spanHatLeft = farPointL - cLeft ;
        spanHatLeft = spanHatLeft' ;
        spanHatLeft = spanHatLeft / norm(spanHatLeft) ;
        
        % makes wing span point outward
        if dot(cLeft-cBody,spanHatLeft) < 0
            spanHatLeft = -spanHatLeft;
        end
        
        
        
        % FIND WING TIP
        % -------------
        
        leftWingTip = findWingTip(leftWingLargestCC, spanHatLeft', cLeft);
        if (isnan(leftWingTip(1)))
            leftWingTip = farPointL ;
        end
        
        % recalculate span vector based on the refined wing tip
        spanHatLeft = leftWingTip - cLeft ;
        spanHatLeft = spanHatLeft' ;
        spanHatLeft = spanHatLeft / norm(spanHatLeft) ;
        
        span2Hat    = spanHatLeft ; % get rid of this duplicity
        
        
        %{
    % RECALC WING CENTER OF MASS AS WINGLENGTH/2 AWAY FROM THE TIP
    cLeft = leftWingTip - span2Hat'*wingLength/2 ;
    newCentroids(leftWingInd,:) = cLeft ;
        %}
        
        % store right wing center of mass (centroid)
        x2s(ind) = cLeft(1) ; % newCentroids(rightWingInd,1) ;
        y2s(ind) = cLeft(2) ; % newCentroids(rightWingInd,2) ;
        z2s(ind) = cLeft(3) ; % newCentroids(rightWingInd,3) ;
        
        % ------------
        %{
    % OLD
    % left wing center of mass (centroid)
    x2s(ind) = cLeft(1) ; % newCentroids(leftWingInd,1) ;
    y2s(ind) = cLeft(2) ; % newCentroids(leftWingInd,2) ;
    z2s(ind) = cLeft(3) ; % newCentroids(leftWingInd,3) ;
    
    % PCA ON LEFT WING VOXELS
    pcaCoeffs      = princomp(coords(leftWingRows,:));
    
    spanHatLeft   = pcaCoeffs(:,1); % 1st axis
    secondHatLeft = pcaCoeffs(:,2); %#ok<NASGU> % 2nd axis
    
    % makes wing span point outward
    if dot(cLeft-cBody,spanHatLeft) < 0
        spanHatLeft = -spanHatLeft;
    end
    
    spanHatLeft = spanHatLeft / norm(spanHatLeft) ;
    span2Hat = spanHatLeft ; % get rid of this duplicity
    
     % ------------
        %}
        
        %{
    % SEE IF THIS PIECE OF CODE IS RELEVANT. IT IS RELATED TO THE CODE IN
    % LINES 594-650 IN THE ORIGINAL FILE
    
    % if wings are obscured at back just take the span to be the vector in
    % the plane of the wing that connects body center to wing center
    if iterClust == 1
        third1Hat = cross(span1Hat,second1Hat);
        third1Hat = third1Hat/norm(third1Hat);
        span1Hat = c1'-cBody' - dot(c1'-cBody',third1Hat)*third1Hat;
        span1Hat = span1Hat/norm(span1Hat);
    end
        %}
        
        % right wing phi and theta.
        % NOTE: here phi is with respect to the x axis in LAB FRAME OF REF.
        phi2   = atan2(spanHatLeft(2),spanHatLeft(1));
        theta2 = asin(spanHatLeft(3));
        
        phi2Proj  = [span2Hat(1),span2Hat(2),0];
        phi2Proj  = phi2Proj / norm(phi2Proj);
        phi2Hat   = [-sin(phi2),cos(phi2),0];
        theta2Hat = -phi2Proj*sin(theta2) + [0,0,1]*cos(theta2);
        
        
        % left wing eta (pitch angle)
        % ----------------------------
        
        try
            if verboseFlag
                disp('Left wing:') ;
            end
            %leftWingVoxels = coords(leftWingRows,:) ;
            leftWingVoxels  = double(leftWingLargestCC) ;
            
            Nvox            = size(leftWingVoxels,1) ;
            
            mat1 = leftWingVoxels - repmat(cLeft, Nvox,1) ;
            mat2 = repmat(spanHatLeft', Nvox, 1) ;
            
            distFromMidSpan = abs(sum(mat1.*mat2,2) ) ;
            clear mat1 mat2
            
            chordRowsInd = find(distFromMidSpan<delta) ;
            
            if (isempty(chordRowsInd))
                % first try a larger delta
                chordRowsInd = find(distFromMidSpan<3*delta) ;
                % check if still empty
                if (isempty(chordRowsInd))
                    error('hullAnalysis:Chord','Bad clustering - empty left chord') ;
                end
            end
            
            clear distFromMidSpan
            
            % among the chordRowsInd voxels, find the pair that is the most distant
            % apart. this pair defines the direction of the chord
            % original code calculated a distance matric between all pairs. here we
            % do it a bit more efficiently by excluding most of the voxels before
            % calculating the distance matrix
            
            % caculate the distance^2 of each voxel from the wing centroid
            Nvox    = length(chordRowsInd);
            mat1    = (leftWingVoxels(chordRowsInd,:) - repmat(cLeft, Nvox,1)).^2 ;
            distVec = (sum(mat1,2)).^0.5 ;
            clear mat1
            
            % select only the top quarter of the voxels, i.e the most distant from
            % wing centroid
            
            [~, sortedInd] = sort(distVec,'descend') ;
            selectedInd    = chordRowsInd(sortedInd(1:ceil(Nvox*chordFraction))) ;
            clear distVec
            
            %selectedIndLeft = selectedInd ;
            % find the most distant pair
            
            distMat = squareform (pdist (leftWingVoxels(selectedInd,:))) ;
            
            [maxRowVec Irow] = max(distMat,[],1) ;
            [~, Icol] = max(maxRowVec) ;
            Irow = Irow(Icol) ;
            
            
            if (distMat(Irow, Icol)~=max(distMat(:))) && verboseFlag
                disp('Error with finding max. plz check.') ;
                %disp('problem 7?') ;
                %keyboard ;
            end
            
            % the following indices give voxel coordinate:
            vox1IndLeft = selectedInd(Irow) ; % voxel position is leftWingVoxels(vox1IndLeft,:)
            vox2IndLeft = selectedInd(Icol) ; % voxel position is leftWingVoxels(vox2IndLeft,:)
            
            chord2Hat = leftWingVoxels(vox1IndLeft,:)' - leftWingVoxels(vox2IndLeft,:)' ;
            
            p1 = leftWingVoxels(vox1IndLeft,:)' ;
            p2 = leftWingVoxels(vox2IndLeft,:)' ;
            
            % force chord vector to be perpendicular to the span vector
            chord2Hat = chord2Hat - span2Hat * dot(span2Hat, chord2Hat) ;
            
            diag21 = norm(chord2Hat) ; % will be used later
            
            chord2Hat = chord2Hat / norm(chord2Hat) ;
            
            if (chord2Hat(3)<0)
                chord2Hat = - chord2Hat ;
            end
            
            % makes chord look good on left wing during upstroke
            
            %disp ('check this vector manipulation');
            % chord2Hat = abs(dot(chord2Hat,phi2Hat))*phi2Hat + abs(dot(chord2Hat,theta2Hat))*theta2Hat;
            
            % FIND THE SECOND DIAGONAL OF THE LEFT WING PARALLELOGRAM (ADDED FEB 2012)
            % ------------------------------------------------------------------------
            
            % see http://mathworld.wolfram.com/Point-PlaneDistance.html
            mat1    = leftWingVoxels(chordRowsInd,:) - repmat(cLeft, Nvox,1) ;
            % mat1 contains the vectors connecting the voxels
            % in chordRowsInd to the wing centoid.
            
            % calc the vector normal to the span and chord
            wingNormVec = cross(span2Hat, chord2Hat) ;
            
            % calc the (signed) distance from each point in mat1 to the wing plane
            mat2 = repmat(wingNormVec', Nvox, 1) ;
            distVec = sum(mat1.*mat2,2) ;
            
            % find the largest positive and largest negative distances, which
            % correspond to the farthest voxels on each side of the plane
            
            [maxval, indmax] = max(distVec) ; % index into rightWingVoxels
            [minval, indmin] = min(distVec) ; % index into rightWingVoxels
            
            if (maxval<=0 || minval>=0) && verboseFlag
                disp('error in finding alternative chord vector for left wing') ;
                %keyboard ;
            end
            
            indmin = chordRowsInd(indmin) ;
            indmax = chordRowsInd(indmax) ;
            chord2AltHat = leftWingVoxels(indmax,:)' - leftWingVoxels(indmin,:)' ;
            
            p3 = leftWingVoxels(indmax,:)' ;
            p4 = leftWingVoxels(indmin,:)' ;
            
            % force alternative chord to be vertical to the span vector
            chord2AltHat = chord2AltHat - span2Hat * dot(span2Hat, chord2AltHat) ;
            
            diag22 = norm(chord2AltHat) ; % will be used later
            
            % now normalize
            chord2AltHat = chord2AltHat / norm(chord2AltHat) ;
            
            % choose the sign of the alt vector
            %if ( dot(chord2AltHat, chord2Hat) < 0 )
            %    chord2AltHat = - chord2AltHat ;
            %end
            
            if (chord2AltHat(3)<0)
                chord2AltHat = - chord2AltHat ;
            end
            
            diag21Left(ind) = diag21 ;
            diag22Left(ind) = diag22 ;
            
            % decide whether to swap the "main" and "alternative" chord vectors
            % -----------------------------------------------------------------
            if verboseFlag
                disp(['Before swap: diag21=' num2str(diag21) ...
                    ' diag22=' num2str(diag22)]) ;
                disp(['diag21/diag22=' num2str(diag21/diag22) ...
                    '   diag22/diag21=' num2str(diag22/diag21)]) ;
                disp(['z coordinates of main chord: ' ...
                    num2str(min([p1(3), p2(3)])) ', ' ...
                    num2str(max([ p1(3), p2(3)]))]) ;
                disp(['z coordinates of alt. chord: ' ...
                    num2str(min([p3(3), p4(3)])) ', ' ...
                    num2str(max([ p3(3), p4(3)]))]) ;
            end
            % if one of the diagonals is siginficantly longer, choose the longer
            % one and do not proceed to the velocity criterion below
            diagSwapFlag     = false ;
            velocitySwapFlag = false ;
            zSwapFlag        = false ;
            
            % for the z-swap - decide to swap if alt-chord has larger z range
            
            if ( max([ p3(3), p4(3)]) > max([ p1(3), p2(3)]) && ...
                    min([ p3(3), p4(3)]) < min([ p1(3), p2(3)]) )
                zSwapFlag = true ;
                if verboseFlag
                    disp('ignored in the meantime: swap based on z range of chords.') ;
                end
            end
            
            if (diag22/diag21 >= 1.3)
                diagSwapFlag = true ;
                if verboseFlag
                    disp('swap based on large ratio') ;
                end
            end
            
            % find wingtip 'velocity' with respect to the body
            if(ind>1)
                % previous version calculate wing centroid velocity:
                vWingCM = [  (x2s(ind) - xs(ind)) - (x2s(ind-1)-xs(ind-1)) ; ...
                    (y2s(ind) - ys(ind)) - (y2s(ind-1)-ys(ind-1))  ; ...
                    (z2s(ind) - zs(ind)) - (z2s(ind-1)-zs(ind-1)) ] ;
                
                vWing =  ( leftWingTip - [xs(ind) ys(ind) zs(ind)] ) - ...
                    ( leftWingTips(ind-1,:) - [xs(ind-1) ys(ind-1) zs(ind-1)]) ;
                
                % keep only the component perpendicular to the span vector
                vWing = vWing - span2Hat' * dot(span2Hat, vWing) ;
                
                nrm = norm(vWing) ;
                if verboseFlag
                    disp(['Wing tip velocity = ' num2str(nrm)]) ;
                    disp(['Wing CM  velocity = ' num2str(norm(vWingCM))]) ;
                end
            else
                nrm = 0 ;
            end
            
            if (ind>1)
                if (nrm~=0)
                    vWing = vWing / nrm ;
                    dot1 = dot(chord2Hat, vWing) ;
                    dot2 = dot(chord2AltHat, vWing) ;
                    if verboseFlag
                        disp(['dot1=' num2str(dot1) '  dot2=' num2str(dot2)]) ;
                    end
                    if (dot2>dot1) % swap
                        velocitySwapFlag = true ;
                    end
                    if (dot1<0 && dot2<0 && nrm>=wingTipVelocityThreshold && ~velocitySwapFlag)
                        if verboseFlag
                            disp('--> inverting left chord. not swapping.') ;
                        end
                        chord2Hat = - chord2Hat ;
                    end
                end
            end
            
            swapFlag = (velocitySwapFlag && nrm>=wingTipVelocityThreshold) || ... % believe velocity if |v|>2
                (diagSwapFlag && nrm<wingTipVelocityThreshold) ;
            
            if (velocitySwapFlag && nrm>=wingTipVelocityThreshold) && verboseFlag
                disp('Left wing chord - velocity swap') ;
            end
            if (diagSwapFlag && nrm<wingTipVelocityThreshold) && verboseFlag
                disp('Left wing chord - diag swap') ;
            end
            
            
            %swapFlag = zSwapFlag || (diagSwapFlag && nrm<wingTipVelocityThreshold) ;
            
            % probably need a smarter way to "weigh" the two types of swaps
            if (swapFlag)
                tmp = chord2Hat ;
                chord2Hat = chord2AltHat ;
                chord2AltHat = tmp ;
                
                tmp = diag21Left(ind) ;
                diag21Left(ind) = diag22Left(ind) ;
                diag22Left(ind) = tmp ;
                clear tmp ;
                if verboseFlag
                    disp('Swapped left chord with alternative chord') ;
                end
            end
            
            
            [p1L, p2L, leftChordTopProj] = chordTopViewProjection(spanHatLeft, cLeft, leftWingLargestCC) ;
            
            % ----------
            
            chord2Hat2 = -abs(dot(chord2Hat,phi2Hat))*phi2Hat + abs(dot(chord2Hat,theta2Hat))*theta2Hat;
            % in the first quadrant only
            eta2 = atan2( abs(dot(chord2Hat,theta2Hat)), abs(dot(chord2Hat,phi2Hat)));
            
        catch %#ok<CTCH>
            if verboseFlag
                disp('error occured while trying to find left chord. taking values from the previous frame.')
            end
            % this ignores the case where the error occures on the first frame...
            vox1IndLeft = [] ;
            vox2IndLeft = [] ;
        end
        
    else
        nv = [NaN NaN NaN] ;
        
        phi1    = NaN ;
        theta1  = NaN ;
        eta1    = NaN ;
        
        phi2    = NaN ;
        theta2  = NaN ;
        eta2    = NaN ;
        
        span1Hat        = nv ;
        chord1Hat       = nv ;
        chord1Hat2      = nv ;
        phi1Hat         = nv ;
        chord1AltHat    = nv ;
        rightWingTip    = nv ;
        rightChordTopProj = NaN ;
        
        span2Hat        = nv ;
        chord2Hat       = nv ;
        chord2Hat2      = nv ;
        phi2Hat         = nv ;
        chord2AltHat    = nv ;
        leftWingTip     = nv ;
        leftChordTopProj = NaN ;
        
    end
    
    % ---------------------------
    % STORE ANGLES FOR BOTH WINGS
    % ---------------------------
    
    phi1s(ind)   = phi1;
    theta1s(ind) = theta1;
    %edited by SW 1/17/2015
    if exist('eta1')
        eta1s(ind)   = eta1 ;
    else
        eta1s(ind)   = NaN ;
    end
    
    phi2s(ind)   = phi2;
    theta2s(ind) = theta2;
    if exist('eta2')
        eta2s(ind)   = eta2 ;
    else
        eta2s(ind)   = NaN ;
    end
    
    % -----------------------------------------
    % STORE VOXELS
    % -----------------------------------------
    
    %bodyPoints{i} = bodyVoxels;
    %wing1Points{i} = wing1Voxels;
    %wing2Points{i} = wing2Voxels;
    
    % -----------------------------------------
    % STORE UNIT VECTORS USED FOR PLOTTING ETC.
    % -----------------------------------------
    rollHats(ind,:) = rollHat' ;
    psiHats(ind,:) = psiHat' ; % used in GUI
    
    span1Hats(ind,:)   = span1Hat' ;
    %edited by SW on 1/17/2015
    if exist('chord1Hat','var')
        chord1Hats(ind,:)  = chord1Hat' ;
    else
        chord1Hats(ind,:) = [NaN NaN NaN] ;
    end
    if exist('chord1Hat2','var')
        chord1Hat2s(ind,:)  = chord1Hat2 ;
    else
        chord1Hat2s(ind,:) = [NaN NaN NaN] ;
    end
    %chord1Hat2s(ind,:) = chord1Hat2 ;
    phi1Hats(ind,:)    = phi1Hat' ;
    
    span2Hats(ind,:)   = span2Hat' ;
    if exist('chord2Hat','var')
        chord2Hats(ind,:)  = chord2Hat' ;
    else
        chord2Hats(ind,:) = [NaN NaN NaN] ;
    end
    %chord2Hats(ind,:)  = chord2Hat' ;
    if exist('chord2Hat2','var')
        chord2Hat2s(ind,:)  = chord2Hat2 ;
    else
        chord2Hat2s(ind,:) = [NaN NaN NaN] ;
    end
    %chord2Hat2s(ind,:) = chord2Hat2 ;
    phi2Hats(ind,:)    = phi2Hat' ;
    
    if exist('chord1AltHat','var')
        chord1AltHats(ind,:)  = chord1AltHat' ;
    else
        chord1AltHats(ind,:) = [NaN NaN NaN] ;
    end
    %chord1AltHats(ind,:) = chord1AltHat' ;
    if exist('chord2AltHat','var')
        chord2AltHats(ind,:)  = chord2AltHat' ;
    else
        chord2AltHats(ind,:) = [NaN NaN NaN] ;
    end
    %chord2AltHats(ind,:) = chord2AltHat' ;
    
    rightWingTips(ind,:) = rightWingTip ;
    leftWingTips(ind,:)  = leftWingTip ;
    
    if exist('rightChordTopProj','var')
        rightChordTopProjections(ind) =  rightChordTopProj ;
    else
        rightChordTopProjections(ind) =  NaN ;
    end
    %rightChordTopProjections(ind) =  rightChordTopProj ;
    if exist('leftChordTopProj','var')
        leftChordTopProjections(ind) =  leftChordTopProj ;
    else
        leftChordTopProjections(ind) =  NaN ;
    end
    %leftChordTopProjections(ind) =  leftChordTopProj ;
    
    if (plotFlag) %&& (ind > 604)
        % plot clustering results
        figure(h) ;
        clf ;
        hold on ;
        %{
        for j=1:K % was (K-1) in the version that used kmeans
            % b = (newIDX==j) ;
            b = RESIDX(i1:i2,j) ;
            plot3(coords(b,1), coords(b,2), coords(b,3),'o', ...
                'markersize',2,'color',colmap(j,:)) ;
            
            plot3(bodyCoords(:,1),bodyCoords(:,2),bodyCoords(:,3),'o','markersize',2,'color',colmap(1,:)) ;
axis equal ; box on ; grid on ;
            
            % scatter3(coords(b,1), coords(b,2), coords(b,3), 6, post(b,postClusterInd(j)),'o') ;
        %{
            x = coords(b,1) ;
            y = coords(b,2) ;
            z = coords(b,3) ;
            p = post(b,postClusterInd(j)) ;
            ip = (p>=probThresh) ;
            scatter3(x(ip), y(ip), z(ip), 6, p(ip),'o') ;
        %}
        end
        %}
        
        plot3(coords(bodyRows,1),coords(bodyRows,2),coords(bodyRows,3),'o','markersize',2,'color',colmap(bodyInd,:)) ;
        plot3(coords(rightWingRows,1),coords(rightWingRows,2),coords(rightWingRows,3),'o','markersize',2,'color',colmap(rightWingInd,:)) ;
        plot3(coords(leftWingRows,1),coords(leftWingRows,2),coords(leftWingRows,3),'o','markersize',2,'color',colmap(leftWingInd,:)) ;
        % plot centroids of three clusters
        plot3(newCentroids(:,1), newCentroids(:,2), newCentroids(:,3), ...
            'ko','markerfacecolor','k','markersize',12) ;
        
        try
            % plot wing tips
            plot3(rightWingTip(1), rightWingTip(2), rightWingTip(3), ...
                'ks','markerfacecolor','k','markersize',10) ;
        catch %#ok<CTCH>
            disp('problem 8?') ;
            %keyboard ;
        end
        
        %{
        % plot chord voxels
        plot3( leftWingVoxels(selectedIndLeft,1),leftWingVoxels(selectedIndLeft,2), ...
            leftWingVoxels(selectedIndLeft,3),'ko') ;
        plot3( rightWingVoxels(selectedIndRight,1),rightWingVoxels(selectedIndRight,2), ...
            rightWingVoxels(selectedIndRight,3),'ko') ;
        %}
        
        A = 60 * (wingLength/45) ;
        B = 30 * (wingLength/45) ;
        % plot body axis
        xvec = [0 A*rollHat(1)] + newCentroids(bodyInd,1) ;
        yvec = [0 A*rollHat(2)] + newCentroids(bodyInd,2) ;
        zvec = [0 A*rollHat(3)] + newCentroids(bodyInd,3) ;
        plot3(xvec, yvec, zvec, 'ko-','linewidth',3,'markersize',8,'markerfacecolor',colmap(bodyInd,:)) ;
        
        % plot right wing span
        xvec = [0 A*span1Hat(1)] + newCentroids(rightWingInd,1) ;
        yvec = [0 A*span1Hat(2)] + newCentroids(rightWingInd,2) ;
        zvec = [0 A*span1Hat(3)] + newCentroids(rightWingInd,3) ;
        plot3(xvec, yvec, zvec, 'ko-','linewidth',3,'markersize',8, 'markerfacecolor',colmap(rightWingInd,:)) ;
        
        % plot right wing chord
        xvec = [0 B*chord1Hat(1)] + newCentroids(rightWingInd,1) ;
        yvec = [0 B*chord1Hat(2)] + newCentroids(rightWingInd,2) ;
        zvec = [0 B*chord1Hat(3)] + newCentroids(rightWingInd,3) ;
        plot3(xvec, yvec, zvec, 'ko-','linewidth',3,'markersize',8, 'markerfacecolor',colmap(rightWingInd,:)) ;
        
        % plot right wing alternative chord
        xvec = [0 B*chord1AltHat(1)] + newCentroids(rightWingInd,1) ;
        yvec = [0 B*chord1AltHat(2)] + newCentroids(rightWingInd,2) ;
        zvec = [0 B*chord1AltHat(3)] + newCentroids(rightWingInd,3) ;
        plot3(xvec, yvec, zvec, 'ks--','linewidth',2,'markersize',8, 'markerfacecolor',colmap(rightWingInd,:)) ;
        
        % plot left wing span
        xvec = [0 A*span2Hat(1)] + newCentroids(leftWingInd,1) ;
        yvec = [0 A*span2Hat(2)] + newCentroids(leftWingInd,2) ;
        zvec = [0 A*span2Hat(3)] + newCentroids(leftWingInd,3) ;
        plot3(xvec, yvec, zvec, 'ko-','linewidth',3,'markersize',8, 'markerfacecolor',colmap(leftWingInd,:)) ;
        
        % plot left wing chord
        xvec = [0 B*chord2Hat(1)] + newCentroids(leftWingInd,1) ;
        yvec = [0 B*chord2Hat(2)] + newCentroids(leftWingInd,2) ;
        zvec = [0 B*chord2Hat(3)] + newCentroids(leftWingInd,3) ;
        plot3(xvec, yvec, zvec, 'ko-','linewidth',3,'markersize',8, 'markerfacecolor',colmap(leftWingInd,:)) ;
        
        % plot left wing alternative chord
        xvec = [0 B*chord2AltHat(1)] + newCentroids(leftWingInd,1) ;
        yvec = [0 B*chord2AltHat(2)] + newCentroids(leftWingInd,2) ;
        zvec = [0 B*chord2AltHat(3)] + newCentroids(leftWingInd,3) ;
        plot3(xvec, yvec, zvec, 'ks--','linewidth',2,'markersize',8, 'markerfacecolor',colmap(rightWingInd,:)) ;
        
        plot3( leftWingVoxels(vox1IndLeft,1), leftWingVoxels(vox1IndLeft,2), ...
            leftWingVoxels(vox1IndLeft,3), 'k^','markerfacecolor','y') ;
        plot3( leftWingVoxels(vox2IndLeft,1), leftWingVoxels(vox2IndLeft,2), ...
            leftWingVoxels(vox2IndLeft,3), 'k^','markerfacecolor','y') ;
        
        plot3( rightWingVoxels(vox1IndRight,1), rightWingVoxels(vox1IndRight,2), ...
            rightWingVoxels(vox1IndRight,3), 'k^','markerfacecolor','y') ;
        plot3( rightWingVoxels(vox2IndRight,1), rightWingVoxels(vox2IndRight,2), ...
            rightWingVoxels(vox2IndRight,3), 'k^','markerfacecolor','y') ;
        
        % plot farthest point for both wings
        
        %[p, ~, list] = farthestPoint(rightWingVoxels, cBody, LL) ;
        
        plot3(farPointRight(1), farPointRight(2), farPointRight(3),'ks','markersize',14,'markerfacecolor','y','linewidth',2) ;
        %plot3(rightWingVoxels(list,1), rightWingVoxels(list,2), rightWingVoxels(list,3),'o','markersize',2,'color',[0.75 0 0]) ;
        plot3(rightWingLargestCC(:,1), rightWingLargestCC(:,2), rightWingLargestCC(:,3),'o','markersize',2,'color',[0.6 0 0]) ;
        
        %[p, ~, list] = farthestPoint(leftWingVoxels, cBody, LL) ;
        plot3(farPointLeft(1), farPointLeft(2), farPointLeft(3),'ks','markersize',14,'markerfacecolor','y','linewidth',2) ;
        %plot3(leftWingVoxels(list,1), leftWingVoxels(list,2), leftWingVoxels(list,3),'o','markersize',2,'color',[0 0 0.75]) ;
        plot3(leftWingLargestCC(:,1), leftWingLargestCC(:,2), leftWingLargestCC(:,3),'o','markersize',2,'color',[0 0 0.6]) ;
        
        %{
        rightWingCoords = coords(rightWingRows,:) ;
        ccolors = hsv(length(clustersRight)) ;
        for j=1:length(clustersRight)
            subCoords = rightWingCoords(clustersRight{j},:) ;
            plot3( subCoords(:,1), subCoords(:,2), subCoords(:,3), ...
                'ko','markerfacecolor',ccolors(j,:)) ;
        end
        %}
        title(['Frame no. ' num2str(ind) ' t=' num2str(t) ' n=' num2str(n) ...
            '  D_R=' num2str(rightChordTopProj) ' D_L=' num2str(leftChordTopProj)]);
        xlabel('x') ; ylabel('y'); zlabel('z') ;
        
        hold off ;
        axis equal ;
        grid on ;
        box on ;
        %{
        % rotate view
        for f=1:1 ;
            view(az,18) ;
            if (f>1) ; pause(0.1) ; end ;
            az=az+0 ;
        end
        %}
        %view(92,8) ;
        view([159 10]);
        %pause (1)  ;
        %%ax = [60 210  0 150 170 260 ] ;
        axis(ax) ;
        %keyboard
        if (~isempty(outputFilename) && saveFigFlag)
            %cd figs ;
            rotate3d on ;
            savefig(gcf,fullfile(saveFigPath, ['frame_' num2str(n) '.fig'])) ;
            %cd ..
            %print(gcf,'-dpng','-r0',['.\figs\fr_' num2str(n) '.png']);
        end
        refresh ;
        %pause(0.05) ;
    end
    
    % display progress and time
    currTime = toc ;
    totalCalcTime = totalCalcTime + toc ;
    if verboseFlag
        disp(['Frame ' num2str(ind) '/' num2str(Nimages) ' took ' ...
            num2str(round(currTime*100)/100) 's. Time so far ' ...
            num2str(round(totalCalcTime*10)/10) 's' ]) ;
        
        %if (ind>=1) ; keyboard ; end ;
        %disp('paused...') ;
        disp(' ') ;
    end
    %if (mergedWingsFlag(n))
    % keyboard ;
    %end
    %pause ;
    
    %{
    catch  err
       % handle exceptions thrown in the main loop
       disp(err);
       errorLog(ind) = true ;
       
       xs(ind) = NaN ;
       ys(ind) = NaN ;
       zs(ind) = NaN ;
       
       x1s(ind)= NaN ;
       y1s(ind)= NaN ;
       z1s(ind)= NaN ;
       
       x2s(ind)= NaN ;
       y2s(ind)= NaN ;
       z2s(ind)= NaN ;
       
    end
    %}
    %keyboard ;
end

% ---------------
% clear variables
% ---------------

clear tempMetaData t spanHatRight spanHatLeft span2Hat span1Hat
clear sortedInd selectedInd secondHatRight secondHatLeft rollHat
clear rightWingVoxels rightWingRows psi phi2Proj
clear phi2 phi1Proj phi1Hat phi1 pcaCoeffs newIDX newCentroids maxmax
clear maxRowVec leftWingVoxels leftWingRows kmeansReplicates
clear j ind i2 i1 eta2 eta1 distMat delta coords clusterSize chordRowInd
clear chord2Hat2 chord2Hat chord1Hat2 chord1Hat cRight cLeft cBody
clear bodyRows beta Nvox K Irow Icol IDX I C phi2Hat
clear f az
clear selectedIndRight selectedIndLeft
clear vox1IndLeft vox1IndRight vox2IndLeft vox2IndRight
clear cTemp rowsTemp d11 d12 d21 d22
% ------------------------------------------------------------------------
% STORE ALL RESULTS IN THE SAME STRUCTURE USED IN THE ORIGINAL VERSION OF
% THE H.R.M.T. CODE
% ------------------------------------------------------------------------

stroke = zeros(Nimages,19);

stroke(:,2) = xs ;
stroke(:,3) = ys ;
stroke(:,4) = zs ;
stroke(:,5) = (180/pi)*psis  ;
stroke(:,6) = (180/pi)*betas ;
stroke(:,7) = (180/pi)*rhos  ;

stroke(:,8)  = x1s ;
stroke(:,9)  = y1s ;
stroke(:,10) = z1s ;
stroke(:,11) = (180/pi)*phi1s   ;
stroke(:,12) = (180/pi)*theta1s ;
stroke(:,13) = (180/pi)*eta1s   ;

stroke(:,14) = x2s ;
stroke(:,15) = y2s ;
stroke(:,16) = z2s ;
stroke(:,17) = (180/pi)*phi2s   ;
stroke(:,18) = (180/pi)*theta2s ;
stroke(:,19) = (180/pi)*eta2s   ;

if (~isempty(outputFilename)) && (saveTempDataFlag)
    save (outputFilename, ...
        'stroke', 'rollHats', 'span1Hats', 'chord1Hats', 'chord1Hat2s', ...
        'phi1Hats', 'span2Hats', 'chord2Hats', 'chord2Hat2s', 'phi2Hats', 'bodyInd', 'rightWingInd', ...
        'psiHats', 'normRolls', 'leftWingInd', 'Nimages','params', 'errorLog', ...
        'chord1AltHats','chord2AltHats','rightWingTips', 'leftWingTips') ;
end

RESIDX = [ RESIDX_BODY ; RESIDX_WING1 ; RESIDX_WING2 ] ;

% sort the res matrix according to time
[res, idxvec] = sortrows(res,1) ;
RESIDX = RESIDX(idxvec,:) ;


data = orderData_mk2 (res, RESIDX, stroke, 0, ...
    chord1Hats, chord2Hats, span1Hats, span2Hats, rollHats, ...
    chord1AltHats, chord2AltHats, ...
    -1, -1, -1, -1, ...
    params, errorLog);

if verboseFlag
    disp('Recalulating body center-of-mass and body-axis') ;
end
[axisHats, midPoints] = findBodyAxis_mk2 (data, false);
data.bodyCM_old = data.bodyCM ;
data.AHat_old   = data.AHat ;
data.AHat = axisHats ;
data.bodyCM = midPoints ;

data.wingLength = wingLength ;


%check to see if left and right wings need to be swapped
df = diff(data.res(:,1)) ;
frameStartInd = [1 ; find(df==1)+1] ;
frameEndInd   = [frameStartInd(2:end)-1 ; size(data.res,1)] ;
clear df ;

for j = 1:Nimages
    
    rightWingCheck = cross(data.AHat(j,:),data.rightSpanHats(j,:)) ;
    leftWingCheck = cross(data.AHat(j,:),data.leftSpanHats(j,:)) ;
    %tempArray1 = sign(rightWingCheck(:,3)) ;
    %tempArray2 = sign(leftWingCheck(:,3)) ; %
    
    if (rightWingCheck(3) > 0) && (leftWingCheck(3) < 0) && (abs(data.AHat(j,3)) < .8)
        % if this is the case, it's likely the wings need to be swapped
        row1 = frameStartInd(j) ;
        row2 = frameEndInd(j) ;
        tempResidx = data.RESIDX(row1:row2, 2) ;
        data.RESIDX(row1:row2, 2) = data.RESIDX(row1:row2, 3) ;
        data.RESIDX(row1:row2, 3) = tempResidx ;
        
        tempChord = data.rightChordHats(j,:) ;
        tempSpan = data.rightSpanHats(j,:) ;
        data.rightChordHats(j,:)= data.leftChordHats(j,:) ;
        data.rightSpanHats(j,:)= data.leftSpanHats(j,:) ;
        data.leftChordHats(j,:) = tempChord ;
        data.leftSpanHats(j,:) = tempSpan ;
        
        tempWingCM = data.rightWingCM(j,:) ;
        data.rightWingCM(j,:) = data.leftWingCM(j,:) ;
        data.leftWingCM(j,:) = tempWingCM ;
        
        data.rightChordTopProjections(j) = leftChordTopProjections(j) ;
        data.leftChordTopProjections(j)  = rightChordTopProjections(j) ;
        data.diag11Right(j) = diag21Left(j) ;
        data.diag12Right(j) = diag22Left(j) ;
        data.diag21Left(j)  = diag11Right(j) ;
        data.diag22Left(j)  = diag12Right(j) ;
        data.rightWingTips(j,:) = leftWingTips(j,:) ;
        data.leftWingTips(j,:)  = rightWingTips(j,:) ;
        
    else
        data.rightChordTopProjections(j) = rightChordTopProjections(j) ;
        data.leftChordTopProjections(j)  = leftChordTopProjections(j) ;
        data.diag11Right(j) = diag11Right(j) ;
        data.diag12Right(j) = diag12Right(j) ;
        data.diag21Left(j)  = diag21Left(j) ;
        data.diag22Left(j)  = diag22Left(j) ;
        data.rightWingTips(j,:) = rightWingTips(j,:) ;
        data.leftWingTips(j,:)  = leftWingTips(j,:) ;
        
    end
    
end

if verboseFlag
    disp('done re-calc') ;
end

%disp('it might me needed to run findBodyAxis TWICE') ;
%keyboard ;
%
% figure ;
% hold on ;
% plot3(xs,ys,zs,'g.-','markerfacecolor','g') ;
% plot3(x1s,y1s,z1s,'r.-','markerfacecolor','r') ;
% plot3(x2s,y2s,z2s,'b.-','markerfacecolor','b') ;
% xlabel('x') ; ylabel('y'); zlabel('z') ;
% grid on
% axis equal ;
% box on ;
% view(3)
% hold off ;

end

% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
% extra functions

function tip = findWingTip(coords, spanVec, wingCM)

% find distances of coords from wingCM
Nvox = size(coords,1) ;

fraction = 0.05 ;

% 1st screening
% -------------
% consider only voxels in a cone of 30 degrees around the line starting
% from wingCM along the direction of spanVec
mat1 = coords - repmat(wingCM, Nvox, 1) ;
mat1 = mat1 ./ repmat( myNorm(mat1), 1,3) ;
mat2 = repmat(spanVec, Nvox, 1) ;
dotprod = dot(mat1, mat2, 2) ;

ind1 = dotprod > 0.5 ; % cos(30 deg) = 0.5

% 2nd screening
% -------------
% take the farthest "fraction" of the voxels and calculate their mean position

coords = coords(ind1,:) ;
Nvox   = size(coords,1) ;

dst  = myNorm ( coords - repmat(wingCM,Nvox,1) ) ;

[~, sortedInd] = sort(dst,'descend') ;
Nvox1 = ceil(Nvox*fraction) ;
selectedInd    = sortedInd(1:Nvox1) ;
if (isempty(selectedInd))
    tip = [NaN NaN NaN] ;
else
    if (numel(selectedInd)==1)
        tip = coords(selectedInd,:) ;
    else
        tip = mean(coords(selectedInd,:)) ;
    end
end

end

% ---------------------------------------------------------------------

function [p, dst, list] = farthestPoint(coords, p0, L)
% finds the point in coords which is farthest from p0
% returns the point coordinate p and its distance from p0 in dst.
% if there are several points with the same distance, return only one.
% also finds the indices of voxels in coords that whose distance from p is
% smaller or equal to L

% find p
Nvox = size(coords,1) ;
mat1 = double(coords) - repmat(p0, Nvox,1) ;
dst2vec  = sum (mat1 .* mat1, 2) ;
[dst ind] = max(dst2vec) ;
p = coords(ind,:) ;

% find list
mat1 = double(coords) - repmat(p, Nvox, 1) ;
dst2vec  = sum (mat1 .* mat1, 2) ;
list = (dst2vec <= L^2) ;

end

% ---------------------------------------------------------------------

function [c xyunique ] = calcCentroidFromTopView(coords, farPoint, L)
% find the centroid of the voxel set "coords" in two steps:
% 0. include only the points whose distnace to the farPoint is more than L
%    L is typically wingLength/3
% 1. consider only the unique (x,y) pairs, ignoing z coordinate. Find their
%    center of mass on the xy plane (xc, yc)
% 2. consider the voxels (x,y,z) for which (x-xc)^2+(y-yc)^2<=delta2.
%    i.e. take only the distance in the xy plane, ignoring their z.
%    Find the center of mass along z of this set only.
%
% retrun the center of wing as well as the unique coordinates in the xy
% plane. this will be used in the function xx to calculate the projection
% of the chord in the xy plane as seen from the top view.

c = zeros(1,3) ;
delta2 = 4 ;

% step 0
% ------
Nvox = size(coords,1) ;
dst = myNorm ( coords - repmat(farPoint,Nvox,1) ) ;
ind0 = dst > L ;

% step 1
% ------
xyunique = unique(coords(ind0,1:2),'rows') ;
cxy      = mean(xyunique) ;
c(1:2)   = cxy ;

% step 2
% ------
coords2 = coords(ind0,:) ;

ind = (coords2(:,1)-c(1)).^2 + (coords2(:,2)-c(2)).^2 <= delta2 ;

c(3) = mean(coords2(ind,3)) ;

end



function [p1 p2 d12] = chordTopViewProjection(spanVec, wingCM, coords)

%[p1 p2 d12] = chordTopViewProjection(spanHatRight, cRight, rightWingLargestCC) ;


% p1 and p2 are the extreme points along the "chord" and d12 is the
% distance between them

plotFlag = false ; % for debugging purposes

delta = 2 ; % width of strip

rcm = double(wingCM(1:2)) ; % work in 2D

sxy = [spanVec(1) spanVec(2) ];   % 2D span vector
sxy = sxy / norm(sxy) ;           % normalize to be a unit vector in xy plane

cxy = [ -spanVec(2) spanVec(1)] ; % rotate spanVec by 90 degrees ("chord" vector)
cxy = cxy / norm(cxy) ;           % normalize to be a unit vector in xy plane

xyunique = unique(coords(:,1:2),'rows') ; % top projection

N = size(xyunique,1) ;

% relative vector w.r.t. wing cm
xyunique = double(xyunique) ;
xyunique(:,1) = xyunique(:,1) - rcm(1) ; % use this shape since it is insensitive
xyunique(:,2) = xyunique(:,2) - rcm(2) ; % to the dimension of rcm (2,1) or (1,2)

% calc the component of each pixel along the sxy direction (sxy is a unit
% vector, so that's fine)
dotprod = dot ( xyunique, repmat(sxy,N,1) ,2) ;

ind = abs(dotprod)<=delta ; % logical indices of pixels inside the delta-strip

clear dotprod ;
% now do the same trick for the pixels in the strip - calculate their
% componenet along the cxy vector

N2 = sum(ind) ; % how many pixels in the strip

if (N2==0)
    disp('no pixels in the delta-strip. assign zero to "chord" xy projection') ;
    p1 = rcm ;
    p2 = rcm ;
    d12 = 0 ;
    disp('problem 9?') ;
    % keyboard ;
    return ;
end

dotprod = dot(xyunique(ind,:), repmat(cxy, N2,1), 2) ;

a1 = max(dotprod) ;
a2 = min(dotprod) ;

if (a1<0 || a2>0)
    disp('problem 10?') ;
    %keyboard ;
end

p1 = rcm + a1 * cxy ;
p2 = rcm + a2 * cxy ;
d12 = norm(p1 - p2) ;

if (plotFlag)
    Lspan = 20 ;
    Lchord = 10 ;
    
    figure(71)
    plot(xyunique(:,1), xyunique(:,2),'b.') ;
    hold on ;
    
    plot(xyunique(ind,1), xyunique(ind,2),'ko','markersize',10) ;
    plot([0 Lspan*sxy(1)], [0 Lspan*sxy(2)],'k-','linewidth',4) ;
    plot([0 Lchord*cxy(1)], [0 Lchord*cxy(2)],'b-','linewidth',4) ;
    
    plot(a1*cxy(1), a1*cxy(2),'k^','markerfacecolor','y','markersize',10) ;
    plot(a2*cxy(1), a2*cxy(2),'k^','markerfacecolor','r','markersize',10) ;
    
    
    axis equal ; box on ; grid on ;
    title(['d_1_2 = ' num2str(d12)]) ;
    hold off ;
end

end
