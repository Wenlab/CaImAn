function [A,C,S,P,YrA] = split_overlapping_neurons(Y,A,b,Cin,fin,S,P,options)

%% refer to merge_components line 107-156. however, A should be fixed. 
%% pay attention to fast_merge and P. 
%% after split, we should extract connected components. 
%% refer to merge_components line 65 graph_connected_comp and line 47-88.
%% refer to matlab bwlabeln. and https://blog.csdn.net/akadiao/article/details/80864835
%% and https://www.cnblogs.com/jins-note/p/9520228.html