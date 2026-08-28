clear;
randpairdir='/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/r_results/permutation_whole_run_residual_rmMotionCSFWM/';
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
for i=1:permnum
    fprintf('%d\n',i);
    p_matrix=zeros(len_l,4);
    % z_matrix=zeros(len_l,4);
    p_matrix(:,1:3)=randpair_meanr_1(:,1:3);
    % z_matrix(:,1:3)=paired_mean(:,1:3);
    for j=1:len_l
        count=0;
        voxelmean=eval(['randpair_meanr_',num2str(i),'(j,4)']);
        for k=1:permnum
            if voxelmean<=voxeldistrib(j,k)
                count=count+1;
            else
                break;
            end
        end
        p_matrix(j,4)=1-count/permnum;
    %     if p_matrix(j,4)==0
    %         z_matrix(j,4)=-sqrt(2)*erfcinv(1/(permnum*2)*2);
    %     elseif p_matrix(j,4)==1
    %         z_matrix(j,4)=-sqrt(2)*erfcinv((1-1/(permnum*2))*2);
    %     else
    %         z_matrix(j,4)=-sqrt(2)*erfcinv(p_matrix(j,4)*2);
    %     end
    end
    pname=[randpairdir,'randpairs_',num2str(i),'/permutation_1000_p.txt'];
    fid=fopen(pname,'w');
    [r,c]=size(p_matrix);
    for m = 1:r
        for n=1:c
            fprintf(fid,'%f\t',p_matrix(m,n));
        end
        fprintf(fid,'\r\n');
    end
    fclose(fid);

end
