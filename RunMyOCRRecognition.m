function [resultTestLabels]=RunMyOCRRecognition(filename, locations, classes)

Files=dir('*.bmp');
features=[];
trainDataLabels=[];
coordinates=[];
counts=[];

%Populate feature matrix and labels for all training images----------------
for k=1:length(Files)
   FileNames=Files(k).name;
   im=imread(FileNames);
   class='adfhkmnopqrsuwxz';
   [tempF,tempCoordinates]=OCR_Extract_Features(im,1);
   counts=[counts;size(tempF,1)];
   x=strfind(class, FileNames(1));
   tempLabels=zeros(size(tempF,1),1);
   for i=1:size(tempF,1)
    tempLabels(i,1)=x;
   end
   features=[features;tempF];
   trainDataLabels=[trainDataLabels;tempLabels];
   coordinates=[coordinates;tempCoordinates];
end

%Normalization-------------------------------------------------------------
meanF=zeros(1,size(features,2));
stdF=zeros(1,size(features,2));

for i=1:size(features,2)
    meanF(1,i)=mean(features(:,i));
    stdF(1,i)=std(features(:,i));
end

normalFeatures=[];

for i=1:size(features,2)
    normalFeatures(:,i)=(features(:,i)-meanF(1,i))/stdF(1,i);
end
%Normalization block ends--------------------------------------------------


%Distance matrix-----------------------------------------------------------
D=dist2(normalFeatures,normalFeatures);
% figure, imagesc(D), title('Distances');
%Distance matrix constructed-----------------------------------------------


[D_sorted,D_index]=sort(D,2);
resultLabels=zeros(size(trainDataLabels,1),1);

%k-NN approach to find labels----------------------------------------------
for i=1:size(trainDataLabels,1)
    temp=D_index(i,1:10);
    tempHist=zeros(1,16);
    for k=1:10
            tempHist(1,trainDataLabels(temp(1,k),1))=tempHist(1,trainDataLabels(temp(1,k),1))+1;
    end
    [M,I]=max(tempHist);
    resultLabels(i,1)=I;
end
%k-NN ends-----------------------------------------------------------------

%Recognition rate on training data-----------------------------------------
counter=0;
for i=1:size(trainDataLabels,1)
    if(resultLabels(i,1)==trainDataLabels(i,1))
        counter=counter+1;
    end
end
disp(sprintf('Training data recognition rate: %f', counter/size(trainDataLabels,1)*100))
%Recognition rate found----------------------------------------------------


% %Individual Recognition rate on training data----------------------------
% counter=0;
% k=1;
% rates=[];
% temps=0;
% for i=1:size(trainDataLabels,1)
%     if(i==temps+counts(k,1))
% %         disp(sprintf('Training data recognition rate for %d: %f', k, counter*100/counts(k,1)));
%         rates=[rates;counter*100/counts(k,1)];
%         temps=temps+counts(k,1);
%         k=k+1;
%         counter=0;
%     end
% %     counter
% %     k
%     if(resultLabels(i,1)==trainDataLabels(i,1))
%         counter=counter+1;
%     end
% end
% rates
% %Recognition rate found--------------------------------------------------

%Write recognised labels on training images--------------------------------
% temp=0;
% for k=1:length(Files)
%    FileNames=Files(k).name;
%    im=imread(FileNames);
%    th=200;im2=im;im2(im<th)=1;im2(im>=th)=0;
%    se=strel('disk',2);im2=imdilate(im2,se);
%    se=strel('disk',1);im2=imerode(im2,se);
%    L=bwlabel(im2);
%    figure, colormap gray, imagesc(~L), title('Connected Components')
%    hold on
%   
%    for i=1:counts(k,1)
%        rectangle('Position',[coordinates(i+temp,4),coordinates(i+temp,2),coordinates(i+temp,3)-coordinates(i+temp,4)+1,coordinates(i+temp,1)-coordinates(i+temp,2)+1], 'EdgeColor','b');
%        text(coordinates(i+temp,3),coordinates(i+temp,2),class(resultLabels(i+temp,1)),'Color','blue','FontSize',16);
%    end
%    temp=temp+counts(k,1);
%    hold off
% end
%Training images labelled with recognised labels---------------------------

conf=ConfusionMatrix(trainDataLabels,resultLabels,16);
% figure, imagesc(conf), title('Confusion Matrix');

testim=imread(filename);

th=200;
testim2=zeros(size(testim,1),size(testim,2));
testim2(testim>th)=0;
testim2(testim<=th)=1;

persistent groundLabels
[testF,groundLabels,coordinates]=OCR_Extract_Features_Test(testim,0,locations,classes);


L1=bwlabel(testim2);
Nc=max(max(L1));

persistent normalTestF
normalTestF=[];

hold on
for i=1:Nc;
        [r,c]=find(L1==i);   
        maxr=max(r);
        minr=min(r);
        maxc=max(c);
        minc=min(c);
        for x=1:size(locations,1)
			if(minr-40<=locations(x,2) && maxr+40>=locations(x,2) && minc-40<=locations(x,1) && maxc+40>=locations(x,1))
                if (maxr-minr)*(maxc-minc)<300
                    continue;
                else
                    text(minc,minr,class(classes(x,1)),'Color','red','FontSize',16);
                end
            end
        end
        			
end
hold off

for i=1:size(features,2)
    normalTestF(:,i)=(testF(:,i)-meanF(1,i))/stdF(1,i);
end

DT=dist2(normalTestF, normalFeatures);
% figure, imagesc(DT), title('Distances');
[DT_sorted,DT_index]=sort(DT,2);
resultTestLabels=zeros(size(normalTestF,1),1);


%k-NN on test image--------------------------------------------------------
for i=1:size(normalTestF,1)
    temp=DT_index(i,1:10);
    tempHist=zeros(1,16);
    for k=1:10
            tempHist(1,trainDataLabels(temp(1,k),1))=tempHist(1,trainDataLabels(temp(1,k),1))+1;
    end
    [M,I]=max(tempHist);
    resultTestLabels(i,1)=I;
end
%k-NN found----------------------------------------------------------------

hold on
for i=1:size(coordinates,1)
    text(coordinates(i,3),coordinates(i,2),class(resultTestLabels(i,1)),'Color','blue','FontSize',16);
end
hold off

%Testing data recognition rate---------------------------------------------
counter1=0;
for i=1:size(groundLabels,1)
    if(resultTestLabels(i,1)==groundLabels(i,1))
        counter1=counter1+1;
    end
end
disp(sprintf('Testing data recognition rate: %f', counter1/size(resultTestLabels,1)*100))
%Recognition rate found----------------------------------------------------

end