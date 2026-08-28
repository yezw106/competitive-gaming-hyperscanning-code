outdir='/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/randpairLists/';
subjects=cell2mat(importdata('/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/subjects.txt'));
pairs=cell2mat(importdata('/sharedata/public/LOL_Project/GameHyperscanning2022/process/r_Game2/pairs.txt'));
pnum=length(pairs(:,1));
pairs_rev=[pairs(:,6:10),pairs(:,1:5)];
pairs_all(1:pnum,1)=string(pairs);
pairs_all((pnum+1):(pnum*2),1)=string(pairs_rev);
count=0;
while count<1000
randindex=randperm(pnum*2);
randsub=subjects(randindex,:);
randpairs=string([randsub(1:pnum,:),randsub((pnum+1):(pnum*2),:)]);
flag=0;
for i=1:pnum
    for j=1:(pnum*2)
        if randpairs(i,1)==pairs_all(j,1)
            flag=1;
            break;
        end
    end
    if flag==1
        break;
    end
end
if flag==1
    continue;
end
count=count+1;
%output
filename=[outdir,'randpairs_',num2str(count),'.txt'];
fid=fopen(filename,'w');
for i=1:pnum
    fprintf(fid,'%s\n',randpairs(i,:));
end
fclose(fid);
end
