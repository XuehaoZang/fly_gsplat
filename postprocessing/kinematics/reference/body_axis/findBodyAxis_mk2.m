function [axisHats,midPoints] = findBodyAxis_mk2 (data, plotFlag)
% Finds the body axis of the fly using the same method used to find the
% wing chord vector. Namely, find the two extreme points of the body that
% have the longest distance between them. Use the body axis found by PCA
% (original algorithm) as a guideline.

if (~exist('plotFlag','var'))
    plotFlag = false ;
end

delta    = round(8 * (data.params.pixPerCM/232)) ; %#ok<NOPRT> % delta=8 for 232 zoom, delta=4 for 118 zoom
fraction = 0.15 ;
colmap   = [0 0.8 0 ; 1 0 0 ; 0 0 1 ];

N = data.Nimages ;

df = diff(data.res(:,1)) ;
frameStartInd = [1 ; find(df==1)+1] ;
frameEndInd   = [frameStartInd(2:end)-1 ; size(data.res,1)] ;
clear df ;

axisHats  = zeros(N,3) ;
midPoints = zeros(N,3) ;
tempAHats = data.AHat ; 

if (plotFlag)
    h=figure ;
else
    hbar = waitbar(0,'Calculating body axis...') ;
end

% increase fraction of yellow voxels
% if we a cap is split into two yellow clusters, take the one who's closer
% to the original AHat

for i=1:N % 1:N
    row1 = frameStartInd(i) ;
    row2 = frameEndInd(i) ;
    coords = data.res(row1:row2,2:4) ; % xyz
    
    IDX = data.RESIDX(row1:row2,:) ;
    
    bodyRows      = (IDX(:,data.bodyInd)==1) ;
    
    bodyVoxels = coords(bodyRows,:) ;
    Nvox       = sum(bodyRows) ;
    
    %{
    % find AHat from pca (don't use the one in 'data' since doing so is not
    % uniquely defined if findBodyAxis.m is called more than once).
    p = princomp(double(bodyVoxels)) ;
    AHat  = p(:,1)' ;
    %AHat = data.AHat(i,:) ;   % body axis vector as found from PCA
    %the lines below might be a little confusing. I'm just going to assume
    %that the fly is upright at first. then, later, check the output of the
    %pca vs the previous ahat. this should allow for beta < 0, which was
    %a problem before
    if ((i < 10) && (AHat(3)<0)) || ((i>=10) && (dot(AHat,axisHats(i-1,:))<0)) 
        AHat = -AHat ;
    end
    %}
    % going to try using previous AHat. This will prevent the axis from
    % being uniquely defined, but findBodyAxis should only be called once
    % anyway
    
    AHat = tempAHats(i,:) ;
    
    Rcm  = data.bodyCM(i,:) ; % body center of mass
    
    % find distances of body voxels from Rcm
    
    mat1 = double(bodyVoxels) - repmat(Rcm, Nvox,1) ;
    
    distFromRcm = ( sum (mat1.^2,2) ) .^ 0.5 ;
    clear mat1
    
    [~, sortedInd] = sort(distFromRcm,'descend') ;
    Nvox1          = ceil(Nvox*fraction) ;
    selectedInd    = sortedInd(1:Nvox1) ;
    
    % separate the voxels that are at the head and tail of the fly. use AHat.
    % if AHat is not avilable then an esitmation for the axis could be found by
    % taking the most distant pair of voxels.
    
    mat1 = double(bodyVoxels(selectedInd,:)) - repmat(Rcm, Nvox1, 1) ;
    mat2 = repmat (AHat, Nvox1, 1) ;
    
    dotprod = sum( mat1 .* mat2 , 2) ;
    
    % find the voxels with distance around the maximum/minimum 
    maxdist = max(dotprod) ;
    mindist = min(dotprod) ;
    
    headFlag = (dotprod>maxdist-delta) ;
    tailFlag = (dotprod<mindist+delta) ;
    
    clear dotprod
    
    % check if the head voxels are in one or more connected component
    % if there's more than one, take the largest
    headVoxels = bodyVoxels(selectedInd(headFlag),:) ;
    headVoxels =  findLargestHullCC (headVoxels) ;
    
    % can do it for the tail too, if needed.
    tailVoxels = bodyVoxels(selectedInd(tailFlag),:) ;
    tailVoxels =  findLargestHullCC (tailVoxels) ;
    
    % find center of mass for head and tail voxels
    %old: headCM = mean (bodyVoxels(selectedInd(headFlag),:)) ;
    %old: tailCM = mean (bodyVoxels(selectedInd(tailFlag),:)) ;
    
    headCM = mean (headVoxels) ;
    tailCM = mean (tailVoxels) ;
    
    midPoints(i,:) = (headCM + tailCM) / 2 ;
    %{
% find the most distant pair

distMat = squareform (pdist (double(bodyVoxels(selectedInd,:)))) ;

[maxRowVec Irow] = max(distMat,[],1) ;
[~, Icol] = max(maxRowVec) ;
Irow = Irow(Icol) ;


if (distMat(Irow, Icol)~=max(distMat(:)))
    disp('Error with finding max. plz check.') ;
    keyboard ;
end


% the following indices give voxel coordinate:
vox1Ind = selectedInd(Irow) ; % voxel position is bodyVoxels(vox1IndRight,:)
vox2Ind = selectedInd(Icol) ; % voxel position is bodyVoxels(vox2IndRight,:)

axisHat = double(bodyVoxels(vox1Ind,:) - bodyVoxels(vox2Ind,:))' ;
axisHat = axisHat / norm(axisHat) ;

if (axisHat(3)<0)
    axisHat = - axisHat ;
end
    %}
    
    axisHat = headCM - tailCM ;
    
    axisHat = axisHat / norm(axisHat) ;
    
    axisHats(i,:) = axisHat ;
    
    if (plotFlag)
        
        try
            figure(h) ;
            clf reset ;
        catch %#ok<CTCH>
            h = figure;
        end
        
        hold on ;
        for j=1:3
            b = data.RESIDX(row1:row2,j) ;
            plot3(coords(b,1), coords(b,2), coords(b,3),'o', ...
                'markersize',2,'color',colmap(j,:)) ;
        end
        
        
        plot3(bodyVoxels(selectedInd,1), bodyVoxels(selectedInd,2), ...
            bodyVoxels(selectedInd,3),'k.') ;
        
        plot3(bodyVoxels(selectedInd(headFlag),1),...
            bodyVoxels(selectedInd(headFlag),2), ...
            bodyVoxels(selectedInd(headFlag),3),'ko','markerfacecolor','y') ;
        
        plot3(bodyVoxels(selectedInd(tailFlag),1),...
            bodyVoxels(selectedInd(tailFlag),2), ...
            bodyVoxels(selectedInd(tailFlag),3),'ko','markerfacecolor','y') ;
        
        A = 60 * (delta/8);
        %B = 30 ;
        % plot old body axis
        xvec = [0 A*AHat(1)] + Rcm(1) ;
        yvec = [0 A*AHat(2)] + Rcm(2) ;
        zvec = [0 A*AHat(3)] + Rcm(3) ;
        plot3(xvec, yvec, zvec, 'ko-','linewidth',4,'markersize',8,'markerfacecolor',colmap(data.bodyInd,:)) ;
        
        % plot new body axis
        %xvec = [ bodyVoxels(vox1Ind,1) bodyVoxels(vox2Ind,1) ] ;
        %yvec = [ bodyVoxels(vox1Ind,2) bodyVoxels(vox2Ind,2) ] ;
        %zvec = [ bodyVoxels(vox1Ind,3) bodyVoxels(vox2Ind,3) ] ;
        
        xvec = [ tailCM(1) headCM(1) ] ;
        yvec = [ tailCM(2) headCM(2) ] ;
        zvec = [ tailCM(3) headCM(3) ] ;
        
        plot3(xvec, yvec, zvec, 'k^--','linewidth',6,'markersize',8,'markerfacecolor','y') ;
        
        title(['Frame no. ' num2str(i) ]) ; % ' t=' num2str(t) ' n=' num2str(n)]);
        xlabel('x') ; ylabel('y'); zlabel('z') ;
        
        hold off ;
        axis equal ;
        axis tight ;
        grid on ;
        box on ; rotate3d on ;
        view(105, 10) ;
        pause(0.1) ; % keyboard ;
    else
        waitbar(i/N,hbar) ;
    end
end

if (~plotFlag)
    close(hbar)
end

end