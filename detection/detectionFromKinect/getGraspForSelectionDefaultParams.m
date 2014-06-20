%%its a two stage cascaded network for detection of graspable region.
%%w1,w2,w_class - 1st satge parameters
%%W1,W2,W_class-2nd stage parameters
%%Author:Tanushree Gupta

% Gets a set of grasps for selected patches in the given input RGB image I,
% depth image D, and depth background DBG. Gives you a nice user-friendly
% interface.
%
% Uses a default set of search parameters to make things easy. 
%
% Arguments:
% I: RGB foreground image (3 channels)
% D: depth foreground image (1 channel)
% DBG: depth background image (1 channel) 

function bestRects = getGraspForSelectionDefaultParams(I,D,DBG)

load('/home/niladri-64/deepGraspingCode_TG/data/bgNums.mat')
load ('/home/niladri-64/deepGraspingCode_TG/data/graspModes24.mat')
load ('/home/niladri-64/deepGraspingCode_TG/weights/graspWFinal.mat')   %%w1,w2,w_class
load ('/home/niladri-64/deepGraspingCode_TG/weights/graspWFinal_2.mat')  %%W1,W2,W_class

load /home/niladri-64/deepGraspingCode_TG/data/graspWhtParams.mat

w_class = w_class(:,1);
W_class = W_class(:,1);   


bestRects = getGraspForSelection(I,D,DBG,w1,w2,w_class,W1,W2,W_class,featMeans,featStds,0:15:(15*11),10:10:90,10:10:90,10,trainModes);
end