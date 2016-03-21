Nc=max(max(L));
figure
imagesc(L)
features=[];
hold on;
for i=1:Nc;
[r,c]=find(L==i);
maxr=max(r);
minr=min(r);
maxc=max(c);
minc=min(c);
rectangle('Position',[minc,minr,maxc-minc+1,maxr-minr+1], 'EdgeColor','w');
cim=im2(minr-1:maxr+1,minc-1:maxc+1);
[centroid, theta, roundness, inmo] = moments(cim, 0);
features=[features;theta,roundness,inmo];
end
hold off