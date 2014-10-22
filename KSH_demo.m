function KSH_demo
%% KSH program of CVPR'12 Supervised Hashing with Kernels
%% modified by weiguo feng ustcrevolutionary@gmail.com


%% load dataset 
trainset = load('../cifar10_train.mat');
testset = load('../cifar10_test.mat');
anchorlbs = load('../anchors_300.mat');
anchorlbs = anchorlbs.anchors;

trainX = trainset.gist';
trainY = trainset.labels + 1;
trainY = int8(trainY);
testX = testset.gist';
testY = testset.labels + 1;
testY = int8(testY);
anchors = trainX(anchorlbs, :);

clear trainset;
clear testset;

[ntrain, d] = size(trainX);
[ntest, ~] = size(testX);
p = 300;
m = p;

% kernel computing
KTrain = sqdist(trainX', anchors');
sigma = mean(mean(KTrain, 2));
KTrain = exp(-KTrain/(2*sigma));
mvec = mean(KTrain);
KTrain = KTrain - repmat(mvec, ntrain, 1);

KTest = sqdist(testX', anchors');
KTest = exp( -KTest / (2*sigma) );
KTest = KTest - repmat(mvec, ntest, 1);

% pairwise label matrix
S0 = -ones(ntrain, ntrain, 'int8');
temp = repmat(trainY,1, ntrain) - repmat(int8(trainY'), ntrain, 1);
temp = find(temp == 0);
S0(temp) = 1;
clear temp;

loopbits = [8, 16, 24, 32, 64];
for r = loopbits
	tic;

	S = single(S0 * int8(r));

	% projection optimization
	KK = KTrain;
	RM = KK'*KK;
	A1 = zeros(m, r);
	flag = zeros(1,r);
	for rr = 1:r
		if rr > 1
			S = S-single(single(y)*single(y'));
		end

		LM = KK'*S*KK;
		[U,V] = eig(LM,RM);
		eigenvalue = diag(V)';
		[eigenvalue,order] = sort(eigenvalue,'descend');
		A1(:,rr) = U(:,order(1));
		tep = A1(:,rr)'*RM*A1(:,rr);
		A1(:,rr) = sqrt(ntrain/tep)*A1(:,rr);
		clear U;    
		clear V;
		clear eigenvalue; 
		clear order; 
		clear tep;  

		[get_vec, cost] = OptProjectionFast(KK, S, A1(:,rr), 500);
		y = single(KK*A1(:,rr)>0);
		ind = find(y <= 0);
		y(ind) = -1;
		clear ind;

		y1 = single(KK*get_vec>0);
		ind = find(y1 <= 0);
		y1(ind) = -1;
		clear ind;
		if y1'*S*y1 > y'*S*y
			flag(rr) = 1;
			A1(:,rr) = get_vec;
			y = y1;
		end
	end


	% encoding 
	traincode = int8(A1'*KTrain' > 0);
	train_time = toc;
	disp(['processed ', num2str(r), 'bit, time: ', num2str(train_time)]);


	testcode = int8(A1'*KTest' > 0);
	traincode = traincode';
	testcode = testcode';

	currfile = sprintf('ksh_%d_50K.mat', r);
	save(currfile, 'traincode', 'testcode', 'A1', 'mvec', 'sigma');

	clear S;
	clear traincode;
	clear testcode;
end

end
