clear all;clc;
%---load the dataset after the pca
dataset_pca_4=load('dataset_pca_4.mat');
dataset_pca_4=dataset_pca_4.X_pca;
%--for 4 dimension
class_mu=zeros(30,5);
class_sigma=zeros(120,5);
test_data=zeros(60,5);
sum1=0;sum2=0;sum3=0;sum4=0;sum5=0;sum6=0;sum7=0;sum8=0;sum9=0;sum10=0;
sum11=0;sum12=0;sum13=0;sum14=0;sum15=0;sum22=0;sum23=0;sum24=0;sum25=0;
sum26=0;sum27=0;sum28=0;sum29=0;sum30=0;sum31=0;sum32=0;sum33=0;sum34=0;
sum35=0;sum36=0;

w1=length(find(dataset_pca_4(:,1)==1))/60;
w2=length(find(dataset_pca_4(:,1)==2))/60;
w3=length(find(dataset_pca_4(:,1)==3))/60;
w4=length(find(dataset_pca_4(:,1)==4))/60;
w5=length(find(dataset_pca_4(:,1)==5))/60;
w6=length(find(dataset_pca_4(:,1)==6))/60;
w7=length(find(dataset_pca_4(:,1)==7))/60;
w8=length(find(dataset_pca_4(:,1)==8))/60;
w9=length(find(dataset_pca_4(:,1)==9))/60;
w10=length(find(dataset_pca_4(:,1)==10))/60;
w11=length(find(dataset_pca_4(:,1)==11))/60;
w12=length(find(dataset_pca_4(:,1)==12))/60;
w13=length(find(dataset_pca_4(:,1)==13))/60;
w14=length(find(dataset_pca_4(:,1)==14))/60;
w15=length(find(dataset_pca_4(:,1)==15))/60;
w22=length(find(dataset_pca_4(:,1)==22))/60;
w23=length(find(dataset_pca_4(:,1)==23))/60;
w24=length(find(dataset_pca_4(:,1)==24))/60;
w25=length(find(dataset_pca_4(:,1)==25))/60;
w26=length(find(dataset_pca_4(:,1)==26))/60;
w27=length(find(dataset_pca_4(:,1)==27))/60;
w28=length(find(dataset_pca_4(:,1)==28))/60;
w29=length(find(dataset_pca_4(:,1)==29))/60;
w30=length(find(dataset_pca_4(:,1)==30))/60;
w31=length(find(dataset_pca_4(:,1)==31))/60;
w32=length(find(dataset_pca_4(:,1)==32))/60;
w33=length(find(dataset_pca_4(:,1)==33))/60;
w34=length(find(dataset_pca_4(:,1)==34))/60;
w35=length(find(dataset_pca_4(:,1)==35))/60;
w36=length(find(dataset_pca_4(:,1)==36))/60;
for num_class=1:15
    %--find dataset of different classes
    dataset_pca_4_num_class=dataset_pca_4...
        (find(dataset_pca_4(:,1)==num_class),2:5);
    %--let the last two data to be the test data and the rest will be the
    %training data
    [row,~]=size(dataset_pca_4_num_class);
    train_data_pca_4=dataset_pca_4_num_class(1:row-2,:);
    test_data_pca_4=dataset_pca_4_num_class(row-1:row,:);
    test_data(2*num_class-1:2*num_class,1)=num_class;  
    test_data(2*num_class-1:2*num_class,2:5)=...
        test_data_pca_4; %--all the test data
    %--copute the mean and covariance of each class
    class_mu_ing=mean(train_data_pca_4);
    class_sigma_ing=cov(train_data_pca_4);
    class_mu(num_class,1)=num_class;   
    class_mu(num_class,2:5)=class_mu_ing;
    class_sigma(num_class*4-3:num_class*4,1)=num_class;
    class_sigma(num_class*4-3:num_class*4,2:5)=class_sigma_ing;
end
for num_class=22:36
    %--find dataset of different classes
    dataset_pca_4_num_class=dataset_pca_4(find(...
        dataset_pca_4(:,1)==num_class),2:5);
    %--let the last two data to be the test data and the rest will be the
    %training data
    [row,~]=size(dataset_pca_4_num_class);
    train_data_pca_4=dataset_pca_4_num_class(1:row-2,:);
    test_data_pca_4=dataset_pca_4_num_class(row-1:row,:);
    test_data(2*num_class-13:2*num_class-12,1)=num_class;  
    test_data(2*num_class-13:2*num_class-12,2:5)=test_data_pca_4; 
    %--all the test data
    %--copute the mean and covariance of each class
    class_mu_ing=mean(train_data_pca_4);
    class_sigma_ing=cov(train_data_pca_4);
    class_mu(num_class-6,1)=num_class;   
    class_mu(num_class-6,2:5)=class_mu_ing;
    class_sigma(4*num_class-27:4*num_class-24,1)=num_class;
    class_sigma(4*num_class-27:4*num_class-24,2:5)=class_sigma_ing;
end
for num_test=1:60
    x=test_data(num_test,2:5);
    %-- build classification function
    g1=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==1),2:5))^0.5)...
        *exp((-0.5)*(x-class_mu(1,2:5))*class_sigma(find(class_sigma...
        (:,1)==1),2:5)^(-1)*(x-class_mu(1,2:5))')*w1;
    g2=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==2),2:5))^0.5...
        )*exp((-0.5)*(x-class_mu(2,2:5))*class_sigma(find(class_sigma(:,...
        1)==2),2:5)^(-1)*(x-class_mu(2,2:5))')*w2;
    g3=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==3),2:5))^0.5)...
        *exp((-0.5)*(x-class_mu(3,2:5))*class_sigma(find(class_sigma(...
        :,1)==3),2:5)^(-1)*(x-class_mu(3,2:5))')*w3;
    g4=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==4),2:5))^...
        0.5)*exp((-0.5)*(x-class_mu(4,2:5))*class_sigma(find(...
        class_sigma(:,1)==4),2:5)^(-1)*(x-class_mu(4,2:5))')*w4;
    g5=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==5),2:5))^...
        0.5)*exp((-0.5)*(x-class_mu(5,2:5))*class_sigma(find(...
        class_sigma(:,1)==5),2:5)^(-1)*(x-class_mu(5,2:5))')*w5;
    g6=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==6),2:5))...
        ^0.5)*exp((-0.5)*(x-class_mu(6,2:5))*class_sigma(find(...
        class_sigma(:,1)==6),2:5)^(-1)*(x-class_mu(6,2:5))')*w6;
    g7=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==7),2:5)...
        )^0.5)*exp((-0.5)*(x-class_mu(7,2:5))*class_sigma(find(...
        class_sigma(:,1)==7),2:5)^(-1)*(x-class_mu(7,2:5))')*w7;
    g8=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==8),2:5)...
        )^0.5)*exp((-0.5)*(x-class_mu(8,2:5))*class_sigma(find(...
        class_sigma(:,1)==8),2:5)^(-1)*(x-class_mu(8,2:5))')*w8;
    g9=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==9),2:5))...
        ^0.5)*exp((-0.5)*(x-class_mu(9,2:5))*class_sigma(find(...
        class_sigma(:,1)==9),2:5)^(-1)*(x-class_mu(9,2:5))')*w9;
    g10=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==10),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(10,2:5))*class_sigma(...
        find(class_sigma(:,1)==10),2:5)^(-1)*(x-class_mu(10,2:5))')*w10;
    g11=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==11),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(11,2:5))*class_sigma(...
        find(class_sigma(:,1)==11),2:5)^(-1)*(x-class_mu(11,2:5))')*w11;
    g12=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==12),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(12,2:5))*class_sigma(...
        find(class_sigma(:,1)==12),2:5)^(-1)*(x-class_mu(12,2:5))')*w12;
    g13=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==13),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(13,2:5))*class_sigma(...
        find(class_sigma(:,1)==13),2:5)^(-1)*(x-class_mu(13,2:5))')*w13;
    g14=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==14),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(14,2:5))*class_sigma(...
        find(class_sigma(:,1)==14),2:5)^(-1)*(x-class_mu(14,2:5))')*w14;
    g15=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==15),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(15,2:5))*class_sigma(...
        find(class_sigma(:,1)==15),2:5)^(-1)*(x-class_mu(15,2:5))')*w15;
    %-- data from 16 to 21 is missing
    g22=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==22),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(16,2:5))*class_sigma(...
        find(class_sigma(:,1)==22),2:5)^(-1)*(x-class_mu(16,2:5))')*w22;
    g23=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==23),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(17,2:5))*class_sigma(...
        find(class_sigma(:,1)==23),2:5)^(-1)*(x-class_mu(17,2:5))')*w23;
    g24=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==24),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(18,2:5))*class_sigma...
        (find(class_sigma(:,1)==24),2:5)^(-1)*(x-class_mu(18,2:5))')*w24;
    g25=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==25),2:5))...
        ^0.5)*exp((-0.5)*(x-class_mu(19,2:5))*class_sigma(find(...
        class_sigma(:,1)==25),2:5)^(-1)*(x-class_mu(19,2:5))')*w25;
    g26=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==26),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(20,2:5))*class_sigma(...
        find(class_sigma(:,1)==26),2:5)^(-1)*(x-class_mu(20,2:5))')*w26;
    g27=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==27),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(21,2:5))*class_sigma(...
        find(class_sigma(:,1)==27),2:5)^(-1)*(x-class_mu(21,2:5))')*w27;
    g28=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==28),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(22,2:5))*class_sigma(...
        find(class_sigma(:,1)==28),2:5)^(-1)*(x-class_mu(22,2:5))')*w28;
    g29=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==29),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(23,2:5))*class_sigma(...
        find(class_sigma(:,1)==29),2:5)^(-1)*(x-class_mu(23,2:5))')*w29;
    g30=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==30),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(24,2:5))*class_sigma(...
        find(class_sigma(:,1)==30),2:5)^(-1)*(x-class_mu(24,2:5))')*w30;
    g31=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==31),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(25,2:5))*class_sigma(...
        find(class_sigma(:,1)==31),2:5)^(-1)*(x-class_mu(25,2:5))')*w31;
    g32=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==32),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(26,2:5))*class_sigma(...
        find(class_sigma(:,1)==32),2:5)^(-1)*(x-class_mu(26,2:5))')*w32;
    g33=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==33),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(27,2:5))*class_sigma(...
        find(class_sigma(:,1)==33),2:5)^(-1)*(x-class_mu(27,2:5))')*w33;
    g34=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==34),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(28,2:5))*class_sigma(...
        find(class_sigma(:,1)==34),2:5)^(-1)*(x-class_mu(28,2:5))')*w34;
    g35=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==35),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(29,2:5))*class_sigma(...
        find(class_sigma(:,1)==35),2:5)^(-1)*(x-class_mu(29,2:5))')*w35;
    g36=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==36),...
        2:5))^0.5)*exp((-0.5)*(x-class_mu(30,2:5))*class_sigma(...
        find(class_sigma(:,1)==36),2:5)^(-1)*(x-class_mu(30,2:5))')*w36;
    result=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g22...
        ,g23,g24,g25,g26,g27,g28,g29,g30,g31,g32,g33,g34,g35,g36];
    [~,index]=max(result);
    if index==1
        sum1=sum1+1;
    elseif index==2
        sum2=sum2+1;
    elseif index==3
        sum3=sum3+1;
    elseif index==4
        sum4=sum4+1;
    elseif index==5
        sum5=sum5+1;
    elseif index==6
        sum6=sum6+1;
    elseif index==7
        sum7=sum7+1;
    elseif index==8
        sum8=sum8+1;
    elseif index==9
        sum9=sum9+1;
    elseif index==10
        sum10=sum10+1;
    elseif index==11
        sum11=sum11+1;
    elseif index==12
        sum12=sum12+1;
    elseif index==13
        sum13=sum13+1;
    elseif index==14
        sum14=sum14+1;
    elseif index==15
        sum15=sum15+1;
    elseif index==16
        sum22=sum22+1;
    elseif index==17
        sum23=sum23+1;
    elseif index==18
        sum24=sum24+1;
    elseif index==19
        sum25=sum25+1;
    elseif index==20
        sum26=sum26+1;
    elseif index==21
        sum27=sum27+1;
    elseif index==22
        sum28=sum28+1;
    elseif index==23
        sum29=sum29+1;
    elseif index==24
        sum30=sum30+1;
    elseif index==25
        sum31=sum31+1;
    elseif index==26
        sum32=sum32+1;
    elseif index==27
        sum33=sum33+1;
    elseif index==28
        sum34=sum34+1;
    elseif index==29
        sum35=sum35+1;
    elseif index==30
        sum36=sum36+1;
    end
end
table=zeros(30,2);
table(1:15,1)=1:15;
table(16:30,1)=22:36;
table(1:30,2)=[sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,...
    sum11,sum12,sum13,sum14,sum15,...
    sum22,sum23,sum24,sum25,sum26,sum27,sum28,sum29,sum30,...
    sum31,sum32,sum33,sum34,sum35,sum36];
 
save('confusion_table_4_PCA.mat','table');




clear all;clc;
%---load the dataset after the pca
dataset_pca_3=load('dataset_3_features.mat');
dataset_pca_3=dataset_pca_3.X_new;
%--for 4 dimension
class_mu=zeros(30,4);
class_sigma=zeros(90,4);
test_data=zeros(60,4);
sum1=0;sum2=0;sum3=0;sum4=0;sum5=0;sum6=0;sum7=0;sum8=0;sum9=0;sum10=0;
sum11=0;sum12=0;sum13=0;sum14=0;sum15=0;sum22=0;sum23=0;sum24=0;...
    sum25=0;sum26=0;
sum27=0;sum28=0;sum29=0;sum30=0;sum31=0;sum32=0;sum33=0;sum34=0;...
    sum35=0;sum36=0;
w1=length(find(dataset_pca_3(:,1)==1))/60;
w2=length(find(dataset_pca_3(:,1)==2))/60;
w3=length(find(dataset_pca_3(:,1)==3))/60;
w4=length(find(dataset_pca_3(:,1)==4))/60;
w5=length(find(dataset_pca_3(:,1)==5))/60;
w6=length(find(dataset_pca_3(:,1)==6))/60;
w7=length(find(dataset_pca_3(:,1)==7))/60;
w8=length(find(dataset_pca_3(:,1)==8))/60;
w9=length(find(dataset_pca_3(:,1)==9))/60;
w10=length(find(dataset_pca_3(:,1)==10))/60;
w11=length(find(dataset_pca_3(:,1)==11))/60;
w12=length(find(dataset_pca_3(:,1)==12))/60;
w13=length(find(dataset_pca_3(:,1)==13))/60;
w14=length(find(dataset_pca_3(:,1)==14))/60;
w15=length(find(dataset_pca_3(:,1)==15))/60;
w22=length(find(dataset_pca_3(:,1)==22))/60;
w23=length(find(dataset_pca_3(:,1)==23))/60;
w24=length(find(dataset_pca_3(:,1)==24))/60;
w25=length(find(dataset_pca_3(:,1)==25))/60;
w26=length(find(dataset_pca_3(:,1)==26))/60;
w27=length(find(dataset_pca_3(:,1)==27))/60;
w28=length(find(dataset_pca_3(:,1)==28))/60;
w29=length(find(dataset_pca_3(:,1)==29))/60;
w30=length(find(dataset_pca_3(:,1)==30))/60;
w31=length(find(dataset_pca_3(:,1)==31))/60;
w32=length(find(dataset_pca_3(:,1)==32))/60;
w33=length(find(dataset_pca_3(:,1)==33))/60;
w34=length(find(dataset_pca_3(:,1)==34))/60;
w35=length(find(dataset_pca_3(:,1)==35))/60;
w36=length(find(dataset_pca_3(:,1)==36))/60;
for num_class=1:15
    %--find dataset of different classes
    dataset_pca_3_num_class=dataset_pca_3(find(...
        dataset_pca_3(:,1)==num_class),2:4);
    %--let the last two data to be the test data and the rest will be the
    %training data
    [row,~]=size(dataset_pca_3_num_class);
    train_data_pca_3=dataset_pca_3_num_class(1:row-2,:);
    test_data_pca_3=dataset_pca_3_num_class(row-1:row,:);
    test_data(2*num_class-1:2*num_class,1)=num_class;  
    test_data(2*num_class-1:2*num_class,2:4)=test_data_pca_3; 
    %--all the test data
    %--copute the mean and covariance of each class
    class_mu_ing=mean(train_data_pca_3);
    class_sigma_ing=cov(train_data_pca_3);
    class_mu(num_class,1)=num_class;   
    class_mu(num_class,2:4)=class_mu_ing;
    class_sigma(num_class*3-2:num_class*3,1)=num_class;
    class_sigma(num_class*3-2:num_class*3,2:4)=class_sigma_ing;
end
for num_class=22:36
    %--find dataset of different classes
    dataset_pca_3_num_class=dataset_pca_3(find(...
        dataset_pca_3(:,1)==num_class),2:4);
    %--let the last two data to be the test data and the rest will be the
    %training data
    [row,~]=size(dataset_pca_3_num_class);
    train_data_pca_3=dataset_pca_3_num_class(1:row-2,:);
    test_data_pca_3=dataset_pca_3_num_class(row-1:row,:);
    test_data(2*num_class-13:2*num_class-12,1)=num_class;  
    test_data(2*num_class-13:2*num_class-12,2:4)=test_data_pca_3; 
    %--all the test data
    %--copute the mean and covariance of each class
    class_mu_ing=mean(train_data_pca_3);
    class_sigma_ing=cov(train_data_pca_3);
    class_mu(num_class-6,1)=num_class;   
    class_mu(num_class-6,2:4)=class_mu_ing;
    class_sigma(3*num_class-18-2:3*num_class-18,1)=num_class;
    class_sigma(3*num_class-18-2:3*num_class-18,2:4)=class_sigma_ing;
end
for num_test=1:60
    x=test_data(num_test,2:4);
    %-- build classification function
    g1=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==1),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(1,2:4))*class_sigma...
        (find(class_sigma(:,1)==1),2:4)^(-1)*(x-class_mu(1,2:4))')*w1;
    g2=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==2),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(2,2:4))*class_sigma(...
        find(class_sigma(:,1)==2),2:4)^(-1)*(x-class_mu(2,2:4))')*w2;
    g3=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==3),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(3,2:4))*class_sigma...
        (find(class_sigma(:,1)==3),2:4)^(-1)*(x-class_mu(3,2:4))')*w3;
    g4=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==4),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(4,2:4))*class_sigma...
        (find(class_sigma(:,1)==4),2:4)^(-1)*(x-class_mu(4,2:4))')*w4;
    g5=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==5),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(5,2:4))*class_sigma...
        (find(class_sigma(:,1)==5),2:4)^(-1)*(x-class_mu(5,2:4))')*w5;
    g6=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==6),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(6,2:4))*class_sigma(...
        find(class_sigma(:,1)==6),2:4)^(-1)*(x-class_mu(6,2:4))')*w6;
    g7=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==7)...
        ,2:4))^0.5)*exp((-0.5)*(x-class_mu(7,2:4))*class_sigma(...
        find(class_sigma(:,1)==7),2:4)^(-1)*(x-class_mu(7,2:4))')*w7;
    g8=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==8)...
        ,2:4))^0.5)*exp((-0.5)*(x-class_mu(8,2:4))*class_sigma(...
        find(class_sigma(:,1)==8),2:4)^(-1)*(x-class_mu(8,2:4))')*w8;
    g9=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==9),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(9,2:4))*class_sigma...
        (find(class_sigma(:,1)==9),2:4)^(-1)*(x-class_mu(9,2:4))')*w9;
    g10=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==10),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(10,2:4))*class_sigma(...
        find(class_sigma(:,1)==10),2:4)^(-1)*(x-class_mu(10,2:4))')*w10;
    g11=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==11),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(11,2:4))*class_sigma...
        (find(class_sigma(:,1)==11),2:4)^(-1)*(x-class_mu(11,2:4))')*w11;
    g12=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==12),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(12,2:4))*class_sigma(...
        find(class_sigma(:,1)==12),2:4)^(-1)*(x-class_mu(12,2:4))')*w12;
    g13=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==13)...
        ,2:4))^0.5)*exp((-0.5)*(x-class_mu(13,2:4))*class_sigma...
        (find(class_sigma(:,1)==13),2:4)^(-1)*(x-class_mu(13,2:4))')*w13;
    g14=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==14),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(14,2:4))*class_sigma(...
        find(class_sigma(:,1)==14),2:4)^(-1)*(x-class_mu(14,2:4))')*w14;
    g15=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==15),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(15,2:4))*class_sigma(...
        find(class_sigma(:,1)==15),2:4)^(-1)*(x-class_mu(15,2:4))')*w15;
    %-- data from 16 to 21 is missing
    g22=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==22),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(16,2:4))*class_sigma(...
        find(class_sigma(:,1)==22),2:4)^(-1)*(x-class_mu(16,2:4))')*w22;
    g23=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==23),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(17,2:4))*class_sigma(...
        find(class_sigma(:,1)==23),2:4)^(-1)*(x-class_mu(17,2:4))')*w23;
    g24=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==24),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(18,2:4))*class_sigma(...
        find(class_sigma(:,1)==24),2:4)^(-1)*(x-class_mu(18,2:4))')*w24;
    g25=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==25),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(19,2:4))*class_sigma(...
        find(class_sigma(:,1)==25),2:4)^(-1)*(x-class_mu(19,2:4))')*w25;
    g26=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==26)...
        ,2:4))^0.5)*exp((-0.5)*(x-class_mu(20,2:4))*class_sigma...
        (find(class_sigma(:,1)==26),2:4)^(-1)*(x-class_mu(20,2:4))')*w26;
    g27=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==27),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(21,2:4))*class_sigma(...
        find(class_sigma(:,1)==27),2:4)^(-1)*(x-class_mu(21,2:4))')*w27;
    g28=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==28),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(22,2:4))*class_sigma...
        (find(class_sigma(:,1)==28),2:4)^(-1)*(x-class_mu(22,2:4))')*w28;
    g29=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==29),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(23,2:4))*class_sigma(...
        find(class_sigma(:,1)==29),2:4)^(-1)*(x-class_mu(23,2:4))')*w29;
    g30=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==30),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(24,2:4))*class_sigma...
        (find(class_sigma(:,1)==30),2:4)^(-1)*(x-class_mu(24,2:4))')*w30;
    g31=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==31),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(25,2:4))*class_sigma...
        (find(class_sigma(:,1)==31),2:4)^(-1)*(x-class_mu(25,2:4))')*w31;
    g32=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==32),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(26,2:4))*class_sigma(...
        find(class_sigma(:,1)==32),2:4)^(-1)*(x-class_mu(26,2:4))')*w32;
    g33=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==33),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(27,2:4))*class_sigma...
        (find(class_sigma(:,1)==33),2:4)^(-1)*(x-class_mu(27,2:4))')*w33;
    g34=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==34)...
        ,2:4))^0.5)*exp((-0.5)*(x-class_mu(28,2:4))*class_sigma...
        (find(class_sigma(:,1)==34),2:4)^(-1)*(x-class_mu(28,2:4))')*w34;
    g35=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==35),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(29,2:4))*class_sigma...
        (find(class_sigma(:,1)==35),2:4)^(-1)*(x-class_mu(29,2:4))')*w35;
    g36=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==36),...
        2:4))^0.5)*exp((-0.5)*(x-class_mu(30,2:4))*class_sigma...
        (find(class_sigma(:,1)==36),2:4)^(-1)*(x-class_mu(30,2:4))')*w36;
    result=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,...
        g22,g23,g24,g25,g26,g27,g28,g29,g30,g31,g32,g33,g34,g35,g36];
    [~,index]=max(result);
    if index==1
        sum1=sum1+1;
    elseif index==2
        sum2=sum2+1;
    elseif index==3
        sum3=sum3+1;
    elseif index==4
        sum4=sum4+1;
    elseif index==5
        sum5=sum5+1;
    elseif index==6
        sum6=sum6+1;
    elseif index==7
        sum7=sum7+1;
    elseif index==8
        sum8=sum8+1;
    elseif index==9
        sum9=sum9+1;
    elseif index==10
        sum10=sum10+1;
    elseif index==11
        sum11=sum11+1;
    elseif index==12
        sum12=sum12+1;
    elseif index==13
        sum13=sum13+1;
    elseif index==14
        sum14=sum14+1;
    elseif index==15
        sum15=sum15+1;
    elseif index==16
        sum22=sum22+1;
    elseif index==17
        sum23=sum23+1;
    elseif index==18
        sum24=sum24+1;
    elseif index==19
        sum25=sum25+1;
    elseif index==20
        sum26=sum26+1;
    elseif index==21
        sum27=sum27+1;
    elseif index==22
        sum28=sum28+1;
    elseif index==23
        sum29=sum29+1;
    elseif index==24
        sum30=sum30+1;
    elseif index==25
        sum31=sum31+1;
    elseif index==26
        sum32=sum32+1;
    elseif index==27
        sum33=sum33+1;
    elseif index==28
        sum34=sum34+1;
    elseif index==29
        sum35=sum35+1;
    elseif index==30
        sum36=sum36+1;
    end
end
table=zeros(30,2);
table(1:15,1)=1:15;
table(16:30,1)=22:36;
table(1:30,2)=[sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10...
    ,sum11,sum12,sum13,sum14,sum15,...
    sum22,sum23,sum24,sum25,sum26,sum27,sum28,sum29,sum30,...
    sum31,sum32,sum33,sum34,sum35,sum36]; 
save('confusion_table_3_feature_selection.mat','table');



clc;clear all;
mu_5_features=load('mu_5_features.mat');
mu_5_features=mu_5_features.mu;
sigma_5_feature=load('Sigma_5_features.mat');
sigma_5_feature=sigma_5_feature.Sigma;
test_data_fs=load('testdata_feature_selection.mat');
test_data_fs=test_data_fs.test_data;
test_data=test_data_fs;
class_sigma=zeros(150,6);
sum1=0;sum2=0;sum3=0;sum4=0;sum5=0;sum6=0;sum7=0;sum8=0;sum9=0;sum10=0;
sum11=0;sum12=0;sum13=0;sum14=0;sum15=0;...
    sum22=0;sum23=0;sum24=0;sum25=0;sum26=0;
sum27=0;sum28=0;sum29=0;sum30=0;sum31=0;...
    sum32=0;sum33=0;sum34=0;sum35=0;sum36=0;
pw=load('pw.mat');
w1=pw.w1;
w2=pw.w2;
w3=pw.w3;
w4=pw.w4;
w5=pw.w5;
w6=pw.w6;
w7=pw.w7;
w8=pw.w8;
w9=pw.w9;
w10=pw.w10;
w11=pw.w11;
w12=pw.w12;
w13=pw.w13;
w14=pw.w14;
w15=pw.w15;
w22=pw.w22;
w23=pw.w23;
w24=pw.w24;
w25=pw.w25;
w26=pw.w26;
w27=pw.w27;
w28=pw.w28;
w29=pw.w29;
w30=pw.w30;
w31=pw.w31;
w32=pw.w32;
w33=pw.w33;
w34=pw.w34;
w35=pw.w35;
w36=pw.w36;
for i=1:15
    sigma_feature=squeeze(sigma_5_feature(i,:,:));
    class_sigma(i*5-4:i*5,1)=i;
    class_sigma(i*5-4:i*5,2:6)=sigma_feature;
end
for i=16:30
    sigma_feature=squeeze(sigma_5_feature(i,:,:));
    class_sigma(i*5-4:i*5,1)=i+6;
    class_sigma(i*5-4:i*5,2:6)=sigma_feature;
end
class_mu=mu_5_features;
for num_test=1:60
    x=test_data(num_test,2:6);
    %-- build classification function
    g1=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==1),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(1,1:5))*class_sigma(...
        find(class_sigma(:,1)==1),2:6)^(-1)*(x-class_mu(1,1:5))')*w1;
    g2=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==2),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(2,1:5))*class_sigma(...
        find(class_sigma(:,1)==2),2:6)^(-1)*(x-class_mu(2,1:5))')*w2;
    g3=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==3)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(3,1:5))*class_sigma(...
        find(class_sigma(:,1)==3),2:6)^(-1)*(x-class_mu(3,1:5))')*w3;
    g4=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==4),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(4,1:5))*class_sigma...
        (find(class_sigma(:,1)==4),2:6)^(-1)*(x-class_mu(4,1:5))')*w4;
    g5=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==5)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(5,1:5))*class_sigma...
        (find(class_sigma(:,1)==5),2:6)^(-1)*(x-class_mu(5,1:5))')*w5;
    g6=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==6)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(6,1:5))*class_sigma...
        (find(class_sigma(:,1)==6),2:6)^(-1)*(x-class_mu(6,1:5))')*w6;
    g7=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==7)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(7,1:5))*class_sigma...
        (find(class_sigma(:,1)==7),2:6)^(-1)*(x-class_mu(7,1:5))')*w7;
    g8=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==8)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(8,1:5))*class_sigma...
        (find(class_sigma(:,1)==8),2:6)^(-1)*(x-class_mu(8,1:5))')*w8;
    g9=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==9)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(9,1:5))*class_sigma...
        (find(class_sigma(:,1)==9),2:6)^(-1)*(x-class_mu(9,1:5))')*w9;
    g10=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==10),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(10,1:5))*class_sigma(...
        find(class_sigma(:,1)==10),2:6)^(-1)*(x-class_mu(10,1:5))')*w10;
    g11=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==11),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(11,1:5))*class_sigma...
        (find(class_sigma(:,1)==11),2:6)^(-1)*(x-class_mu(11,1:5))')*w11;
    g12=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==12),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(12,1:5))*class_sigma...
        (find(class_sigma(:,1)==12),2:6)^(-1)*(x-class_mu(12,1:5))')*w12;
    g13=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==13)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(13,1:5))*class_sigma(...
        find(class_sigma(:,1)==13),2:6)^(-1)*(x-class_mu(13,1:5))')*w13;
    g14=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==14),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(14,1:5))*class_sigma...
        (find(class_sigma(:,1)==14),2:6)^(-1)*(x-class_mu(14,1:5))')*w14;
    g15=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==15),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(15,1:5))*class_sigma...
        (find(class_sigma(:,1)==15),2:6)^(-1)*(x-class_mu(15,1:5))')*w15;
    %-- data from 16 to 21 is missing
    g22=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==22),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(16,1:5))*class_sigma(...
        find(class_sigma(:,1)==22),2:6)^(-1)*(x-class_mu(16,1:5))')*w22;
    g23=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==23),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(17,1:5))*class_sigma(...
        find(class_sigma(:,1)==23),2:6)^(-1)*(x-class_mu(17,1:5))')*w23;
    g24=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==24),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(18,1:5))*class_sigma(...
        find(class_sigma(:,1)==24),2:6)^(-1)*(x-class_mu(18,1:5))')*w24;
    g25=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==25),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(19,1:5))*class_sigma...
        (find(class_sigma(:,1)==25),2:6)^(-1)*(x-class_mu(19,1:5))')*w25;
    g26=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==26)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(20,1:5))*class_sigma...
        (find(class_sigma(:,1)==26),2:6)^(-1)*(x-class_mu(20,1:5))')*w26;
    g27=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==27)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(21,1:5))*class_sigma...
        (find(class_sigma(:,1)==27),2:6)^(-1)*(x-class_mu(21,1:5))')*w27;
    g28=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==28)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(22,1:5))*class_sigma...
        (find(class_sigma(:,1)==28),2:6)^(-1)*(x-class_mu(22,1:5))')*w28;
    g29=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==29)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(23,1:5))*class_sigma...
        (find(class_sigma(:,1)==29),2:6)^(-1)*(x-class_mu(23,1:5))')*w29;
    g30=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==30),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(24,1:5))*class_sigma(...
        find(class_sigma(:,1)==30),2:6)^(-1)*(x-class_mu(24,1:5))')*w30;
    g31=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==31)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(25,1:5))*class_sigma...
        (find(class_sigma(:,1)==31),2:6)^(-1)*(x-class_mu(25,1:5))')*w31;
    g32=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==32),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(26,1:5))*class_sigma...
        (find(class_sigma(:,1)==32),2:6)^(-1)*(x-class_mu(26,1:5))')*w32;
    g33=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==33),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(27,1:5))*class_sigma(...
        find(class_sigma(:,1)==33),2:6)^(-1)*(x-class_mu(27,1:5))')*w33;
    g34=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==34)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(28,1:5))*class_sigma(...
        find(class_sigma(:,1)==34),2:6)^(-1)*(x-class_mu(28,1:5))')*w34;
    g35=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==35)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(29,1:5))*class_sigma...
        (find(class_sigma(:,1)==35),2:6)^(-1)*(x-class_mu(29,1:5))')*w35;
    g36=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==36)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(30,1:5))*class_sigma...
        (find(class_sigma(:,1)==36),2:6)^(-1)*(x-class_mu(30,1:5))')*w36;
    result=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,...
        g22,g23,g24,g25,g26,g27,g28,g29,g30,g31,g32,g33,g34,g35,g36];
    [~,index]=max(result);
    if index==1
        sum1=sum1+1;
    elseif index==2
        sum2=sum2+1;
    elseif index==3
        sum3=sum3+1;
    elseif index==4
        sum4=sum4+1;
    elseif index==5
        sum5=sum5+1;
    elseif index==6
        sum6=sum6+1;
    elseif index==7
        sum7=sum7+1;
    elseif index==8
        sum8=sum8+1;
    elseif index==9
        sum9=sum9+1;
    elseif index==10
        sum10=sum10+1;
    elseif index==11
        sum11=sum11+1;
    elseif index==12
        sum12=sum12+1;
    elseif index==13
        sum13=sum13+1;
    elseif index==14
        sum14=sum14+1;
    elseif index==15
        sum15=sum15+1;
    elseif index==16
        sum22=sum22+1;
    elseif index==17
        sum23=sum23+1;
    elseif index==18
        sum24=sum24+1;
    elseif index==19
        sum25=sum25+1;
    elseif index==20
        sum26=sum26+1;
    elseif index==21
        sum27=sum27+1;
    elseif index==22
        sum28=sum28+1;
    elseif index==23
        sum29=sum29+1;
    elseif index==24
        sum30=sum30+1;
    elseif index==25
        sum31=sum31+1;
    elseif index==26
        sum32=sum32+1;
    elseif index==27
        sum33=sum33+1;
    elseif index==28
        sum34=sum34+1;
    elseif index==29
        sum35=sum35+1;
    elseif index==30
        sum36=sum36+1;
    end
end
table=zeros(30,2);
table(1:15,1)=1:15;
table(16:30,1)=22:36;
table(1:30,2)=[sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,...
    sum9,sum10,sum11,sum12,sum13,sum14,sum15,...
    sum22,sum23,sum24,sum25,sum26,sum27,sum28,sum29,...
    sum30,sum31,sum32,sum33,sum34,sum35,sum36]; 
save('confusion_table_5_bayesian_fs.mat','table');



clc;clear all;
mu_5_features=load('mu_5_pca.mat');
mu_5_features=mu_5_features.mu;
sigma_5_feature=load('Sigma_5_pca.mat');
sigma_5_feature=sigma_5_feature.Sigma;
test_data_fs=load('testdata_PCA.mat');
test_data_fs=test_data_fs.test_data;
test_data=test_data_fs;
class_sigma=zeros(150,6);
sum1=0;sum2=0;sum3=0;sum4=0;sum5=0;sum6=0;sum7=0;sum8=0;sum9=0;sum10=0;
sum11=0;sum12=0;sum13=0;sum14=0;sum15=0;
sum22=0;sum23=0;sum24=0;sum25=0;sum26=0;
sum27=0;sum28=0;sum29=0;sum30=0;sum31=0;
sum32=0;sum33=0;sum34=0;sum35=0;sum36=0;
pw=load('pw.mat');
w1=pw.w1;
w2=pw.w2;
w3=pw.w3;
w4=pw.w4;
w5=pw.w5;
w6=pw.w6;
w7=pw.w7;
w8=pw.w8;
w9=pw.w9;
w10=pw.w10;
w11=pw.w11;
w12=pw.w12;
w13=pw.w13;
w14=pw.w14;
w15=pw.w15;
w22=pw.w22;
w23=pw.w23;
w24=pw.w24;
w25=pw.w25;
w26=pw.w26;
w27=pw.w27;
w28=pw.w28;
w29=pw.w29;
w30=pw.w30;
w31=pw.w31;
w32=pw.w32;
w33=pw.w33;
w34=pw.w34;
w35=pw.w35;
w36=pw.w36;
for i=1:15
    sigma_feature=squeeze(sigma_5_feature(i,:,:));
    class_sigma(i*5-4:i*5,1)=i;
    class_sigma(i*5-4:i*5,2:6)=sigma_feature;
end
for i=16:30
    sigma_feature=squeeze(sigma_5_feature(i,:,:));
    class_sigma(i*5-4:i*5,1)=i+6;
    class_sigma(i*5-4:i*5,2:6)=sigma_feature;
end
class_mu=mu_5_features;
for num_test=1:60
    x=test_data(num_test,2:6);
    %-- build classification function
    g1=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==1),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(1,1:5))*class_sigma(...
        find(class_sigma(:,1)==1),2:6)^(-1)*(x-class_mu(1,1:5))')*w1;
    g2=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==2)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(2,1:5))*class_sigma...
        (find(class_sigma(:,1)==2),2:6)^(-1)*(x-class_mu(2,1:5))')*w2;
    g3=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==3),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(3,1:5))*class_sigma...
        (find(class_sigma(:,1)==3),2:6)^(-1)*(x-class_mu(3,1:5))')*w3;
    g4=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==4)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(4,1:5))*class_sigma...
        (find(class_sigma(:,1)==4),2:6)^(-1)*(x-class_mu(4,1:5))')*w4;
    g5=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==5),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(5,1:5))*class_sigma...
        (find(class_sigma(:,1)==5),2:6)^(-1)*(x-class_mu(5,1:5))')*w5;
    g6=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==6),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(6,1:5))*class_sigma...
        (find(class_sigma(:,1)==6),2:6)^(-1)*(x-class_mu(6,1:5))')*w6;
    g7=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==7),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(7,1:5))*class_sigma...
        (find(class_sigma(:,1)==7),2:6)^(-1)*(x-class_mu(7,1:5))')*w7;
    g8=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==8),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(8,1:5))*class_sigma...
        (find(class_sigma(:,1)==8),2:6)^(-1)*(x-class_mu(8,1:5))')*w8;
    g9=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==9),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(9,1:5))*class_sigma...
        (find(class_sigma(:,1)==9),2:6)^(-1)*(x-class_mu(9,1:5))')*w9;
    g10=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==10),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(10,1:5))*class_sigma...
        (find(class_sigma(:,1)==10),2:6)^(-1)*(x-class_mu(10,1:5))')*w10;
    g11=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==11),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(11,1:5))*class_sigma(...
        find(class_sigma(:,1)==11),2:6)^(-1)*(x-class_mu(11,1:5))')*w11;
    g12=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==12),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(12,1:5))*class_sigma(...
        find(class_sigma(:,1)==12),2:6)^(-1)*(x-class_mu(12,1:5))')*w12;
    g13=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==13)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(13,1:5))*class_sigma(...
        find(class_sigma(:,1)==13),2:6)^(-1)*(x-class_mu(13,1:5))')*w13;
    g14=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==14),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(14,1:5))*class_sigma(...
        find(class_sigma(:,1)==14),2:6)^(-1)*(x-class_mu(14,1:5))')*w14;
    g15=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==15),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(15,1:5))*class_sigma(...
        find(class_sigma(:,1)==15),2:6)^(-1)*(x-class_mu(15,1:5))')*w15;
    %-- data from 16 to 21 is missing    
    g22=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==22),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(16,1:5))*class_sigma...
        (find(class_sigma(:,1)==22),2:6)^(-1)*(x-class_mu(16,1:5))')*w22;
    g23=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==23),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(17,1:5))*class_sigma(...
        find(class_sigma(:,1)==23),2:6)^(-1)*(x-class_mu(17,1:5))')*w23;
    g24=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==24),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(18,1:5))*class_sigma...
        (find(class_sigma(:,1)==24),2:6)^(-1)*(x-class_mu(18,1:5))')*w24;
    g25=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==25),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(19,1:5))*class_sigma(...
        find(class_sigma(:,1)==25),2:6)^(-1)*(x-class_mu(19,1:5))')*w25;
    g26=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==26),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(20,1:5))*class_sigma(...
        find(class_sigma(:,1)==26),2:6)^(-1)*(x-class_mu(20,1:5))')*w26;
    g27=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==27),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(21,1:5))*class_sigma(...
        find(class_sigma(:,1)==27),2:6)^(-1)*(x-class_mu(21,1:5))')*w27;
    g28=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==28)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(22,1:5))*class_sigma(...
        find(class_sigma(:,1)==28),2:6)^(-1)*(x-class_mu(22,1:5))')*w28;
    g29=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==29),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(23,1:5))*class_sigma...
        (find(class_sigma(:,1)==29),2:6)^(-1)*(x-class_mu(23,1:5))')*w29;
    g30=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==30),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(24,1:5))*class_sigma(...
        find(class_sigma(:,1)==30),2:6)^(-1)*(x-class_mu(24,1:5))')*w30;
    g31=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==31),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(25,1:5))*class_sigma...
        (find(class_sigma(:,1)==31),2:6)^(-1)*(x-class_mu(25,1:5))')*w31;
    g32=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==32)...
        ,2:6))^0.5)*exp((-0.5)*(x-class_mu(26,1:5))*class_sigma...
        (find(class_sigma(:,1)==32),2:6)^(-1)*(x-class_mu(26,1:5))')*w32;
    g33=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==33),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(27,1:5))*class_sigma...
        (find(class_sigma(:,1)==33),2:6)^(-1)*(x-class_mu(27,1:5))')*w33;
    g34=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==34),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(28,1:5))*class_sigma...
        (find(class_sigma(:,1)==34),2:6)^(-1)*(x-class_mu(28,1:5))')*w34;
    g35=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==35),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(29,1:5))*class_sigma(...
        find(class_sigma(:,1)==35),2:6)^(-1)*(x-class_mu(29,1:5))')*w35;
    g36=1/((2*pi)^2*det(class_sigma(find(class_sigma(:,1)==36),...
        2:6))^0.5)*exp((-0.5)*(x-class_mu(30,1:5))*class_sigma(...
        find(class_sigma(:,1)==36),2:6)^(-1)*(x-class_mu(30,1:5))')*w36;
    result=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15...
        ,g22,g23,g24,g25,g26,g27,g28,g29,g30,g31,g32,g33,g34,g35,g36];
    [~,index]=max(result);
    if index==1
        sum1=sum1+1;
    elseif index==2
        sum2=sum2+1;
    elseif index==3
        sum3=sum3+1;
    elseif index==4
        sum4=sum4+1;
    elseif index==5
        sum5=sum5+1;
    elseif index==6
        sum6=sum6+1;
    elseif index==7
        sum7=sum7+1;
    elseif index==8
        sum8=sum8+1;
    elseif index==9
        sum9=sum9+1;
    elseif index==10
        sum10=sum10+1;
    elseif index==11
        sum11=sum11+1;
    elseif index==12
        sum12=sum12+1;
    elseif index==13
        sum13=sum13+1;
    elseif index==14
        sum14=sum14+1;
    elseif index==15
        sum15=sum15+1;
    elseif index==16
        sum22=sum22+1;
    elseif index==17
        sum23=sum23+1;
    elseif index==18
        sum24=sum24+1;
    elseif index==19
        sum25=sum25+1;
    elseif index==20
        sum26=sum26+1;
    elseif index==21
        sum27=sum27+1;
    elseif index==22
        sum28=sum28+1;
    elseif index==23
        sum29=sum29+1;
    elseif index==24
        sum30=sum30+1;
    elseif index==25
        sum31=sum31+1;
    elseif index==26
        sum32=sum32+1;
    elseif index==27
        sum33=sum33+1;
    elseif index==28
        sum34=sum34+1;
    elseif index==29
        sum35=sum35+1;
    elseif index==30
        sum36=sum36+1;
    end
end
table=zeros(30,2);
table(1:15,1)=1:15;
table(16:30,1)=22:36;
table(1:30,2)=[sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8...
    ,sum9,sum10,sum11,sum12,sum13,sum14,sum15,...
    sum22,sum23,sum24,sum25,sum26,sum27,sum28,sum29...
    ,sum30,sum31,sum32,sum33,sum34,sum35,sum36];
 
save('confusion_table_5_bayesian_pca.mat','table');
