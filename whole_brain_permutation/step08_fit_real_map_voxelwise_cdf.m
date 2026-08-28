clear;
randpairdir='/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/r_results/permutation_whole_run_residual_rmMotionCSFWM/';
paireddir='/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/r_results/paired_whole_run_residual_rmMotionCSFWM/';
paired_mean=load([paireddir,'paired_meanr.txt']);
outdir='/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/r_results/permutation_whole_run_residual_rmMotionCSFWM/';

permnum=1000;
for i=1:permnum
    fprintf('%d\n',i);
    filename=[randpairdir,'randpairs_',num2str(i),'/randpair_meanr_',num2str(i),'.txt'];
    eval(['randpair_meanr_',num2str(i),'=load(''',filename,''');']);
end
len_l=length(randpair_meanr_1(:,1));
voxeldistrib=zeros(len_l,1000);
for i=1:permnum
    voxeldistrib(:,i)=eval(['randpair_meanr_',num2str(i),'(:,4);']);
end
voxeldistrib=sort(voxeldistrib,2,'descend');

% fit dist for each voxel
voxeldistrib_fit=prob.NormalDistribution(len_l,1);
for i=1:len_l
    voxeldistrib_fit(i,1)=fitdist(voxeldistrib(i,:)','normal');
end

%get p from distribution_fit for each voxel
p_matrix=zeros(len_l,4);
p_matrix(:,1:3)=paired_mean(:,1:3);
for i=1:len_l
    p_matrix(i,4)=cdf(voxeldistrib_fit(i,1),paired_mean(i,4));
end

p_matrix_scaled=p_matrix;
p_matrix_scaled(:,4)=p_matrix(:,4)-0.999;
p_matrix_scaled(p_matrix_scaled(:,4)<0,4)=0;
p_matrix_scaled(:,4)=p_matrix_scaled(:,4)*10000;

pname=[outdir,'permutation_1000_p_fit_threshold0.001_scale10000.txt'];
fid=fopen(pname,'w');
[r,c]=size(p_matrix_scaled);
for m = 1:r
    for n=1:c
        fprintf(fid,'%f\t',p_matrix_scaled(m,n));
    end
    fprintf(fid,'\r\n');
end
fclose(fid);
