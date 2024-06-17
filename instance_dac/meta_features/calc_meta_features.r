library(Rcatch22)

# Calcualting raw meta-representations for Sigmoid benchmark
# Input data should be eval.csv from Sigmpoid benchmark
# Matrix M consists of the calcualte meta-representations
data<-read.csv(file.choose())

calcualte_raw_sigmoid_features<-function(data){
	data<-data[data$test_set_id=="sigmoid_2D3M_train.csv",]
	episodes<-unique(data$episode)
	test_set<-unique(data$test_set_id)
	seeds<-unique(data$seed)
	M<-matrix(0,length(episodes)*length(seeds),37)
	counter<-1
	for(i in 1:length(episodes)){
		data_temp<-data[data$episode==episodes[i],]
		for(j in 1:length(seeds)){
			data_k<-data_temp[data_temp$seed==seeds[j],]
			temp<-c(unlist((data_k[5]+1)/5),unlist((data_k[6]+1)/10),unlist(data_k[12]),unlist(data_k[1,c(8,9,10,11)]))
			M[counter,1]<-data_k[1,1]
			M[counter,2]<-data_k[1,4]
			M[counter,3]<-data_k[1,3]
			M[counter,4:37]<-temp
			counter<-counter+1
		
		
		}
	}
	colnames(M)<-c("episode","seed","instance",names(temp))

	return(M)
}





# Calcualting catch22 meta-representations for Sigmoid benchmark
# Input data should be eval.csv from Sigmpoid benchmark
# Matrix M consists of the calcualte meta-representations

calcualte_catch22_sigmoid_features<-function(data){
	data<-data[data$test_set_id=="sigmoid_2D3M_train.csv",]
	episodes<-unique(data$episode)
	test_set<-unique(data$test_set_id)
	seeds<-unique(data$seed)
	M<-matrix(0,length(episodes)*length(seeds),79)
	counter<-1
	for(i in 1:length(episodes)){
		data_temp<-data[data$episode==episodes[i],]
		for(j in 1:length(seeds)){
			data_k<-data_temp[data_temp$seed==seeds[j],]
			data_k<-data_k[data_k$test_set_id=="sigmoid_2D3M_train.csv",]
		
			state_1<-catch22_all(unlist((data_k[5]+1)/5),catch24=TRUE)
			state_2<-catch22_all(unlist((data_k[6]+1)/10),catch24=TRUE)
			reward<-catch22_all(unlist(data_k[12]),catch24=TRUE)

		
			temp<-c(state_1[,2],state_2[,2],reward[,2],unlist(data_k[1,c(8,9,10,11)]))
			M[counter,1]<-data_k[1,1]
			M[counter,2]<-data_k[1,4]
			M[counter,3]<-data_k[1,3]
			M[counter,4:79]<-temp
			counter<-counter+1
		
		
		}
	}
	s1<-c()
	for(i in 1:length(state_1[,1]))
	{
		s1[i]<-paste("S1_",state_1[,1][i],sep="",collapse="")
	}
	s2<-c()
	for(i in 1:length(state_2[,1]))
	{
		s2[i]<-paste("S2_",state_2[,1][i],sep="",collapse="")
	}
	r<-c()
	for(i in 1:length(reward[,1]))
	{
		r[i]<-paste("R_",reward[,1][i],sep="",collapse="")
	}
	colnames(M)<-c("episode","seed","instance",s1,s2,r,names(temp)[73:76])
	return(M)
}


# Calcualting catch22 meta-representations for Sigmoid benchmark
# Input data should be eval.csv from CMA-ES and BBOB benchmark
# Matrix M consists of the calcualte meta-representations
data<-read.csv(file.choose())
calcualte_catch22_CMAES_features<-function(data){
	episodes<-unique(data$episode)
	test_set<-unique(data$test_set_id)
	seeds<-unique(data$seed)
	instances<-unique(data$instance)

	counter<-1
	M<-matrix(0,100000,51)
	for(i in 1:length(seeds)){
	
	
		data_temp<-data[data$seed==seeds[i],]
	
		for(j in 1:length(instances)){
			data_k<-data_temp[data_temp$instance==instances[j],]
			for (k in unique(data_k$episode)){
				data_u<-data_k[data_k$episode==k,]
			
				action<-catch22_all(data_u$action_0,catch24=TRUE)
				reward<-catch22_all(data_u$reward,catch24=TRUE)
				temp<-c(action[,2],reward[,2])
				temp<-c(k,instances[j],seeds[i],temp)
				M[counter,]<-temp
				counter<-counter+1
				}
		
		
		}
	
	}

	s2<-c()
	for(i in 1:length(action[,1]))
	{
		s2[i]<-paste("A_",action[,1][i],sep="",collapse="")
	}
	r<-c()
	for(i in 1:length(reward[,1]))
	{
		r[i]<-paste("R_",reward[,1][i],sep="",collapse="")
	}
	colnames(M)<-c("episode","instance","seed",s2,r)
	return(M)
}

