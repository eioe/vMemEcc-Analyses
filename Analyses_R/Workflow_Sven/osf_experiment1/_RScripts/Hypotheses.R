# Demonstrate parameters

pdf(paste(path_figures,"Figure1b.pdf",sep=""), width = 10/2.54, height = 6.5/2.54)
par(mfrow=c(1,2),mar=c(3.5,3.5,0.1,0.1),ps=7,cex=1) # Fontsize of 7

#df <- data.frame(Asym=c(0.7,0.8,0.9,rep(c(0.8),time=6)),xmid=c(0.6,0.6,0.6,0.5,0.6,0.7,0.6,0.6,0.6),scal=c(-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,0.2,-0.2,-0.6),colors=rep(c("black","darkgrey","grey"),times=3))
#df$colors <- as.character(df$colors)


df <- data.frame(Asym=rep(0.76,times=6),xmid=c(0.55,0.55,0.55,0.9,0.65,0.55),scal=c(0.8,0.9,0.8,0.8,0.8,0.8),colors=rep(c("black",defblue,deforange),times=2))
df$colors <- as.character(df$colors)

x <- seq(0.1,3.2,length.out=100)
for(i in 1:6){
  y <- df$Asym[i]+(df$xmid[i]-df$Asym[i])*exp(-exp(df$scal[i])*x)
  if(i == 2 | i ==4){
    plot(x,y,xlim=c(-0.5,3.2),ylim=c(0.45,1),xlab=c("Cue/mask delays [ms]"),ylab=c("Proportion correct"),bty="n",col=df$colors[i],type="l",lwd=1,xaxt="n",yaxt="n")	
    axis(side=1,at = c(0.1,1,3.2),labels=c("100","1000","3200"),tck=-0.02,lwd=0.5)
    axis(side=2,tck=-0.02,lwd=0.5)    
  }
  if(i > 1){
    lines(x,y,col=df$colors[i],lwd=2)	
  }
}
#dev.off()

y1  <- df$Asym[5]+(df$xmid[5]-df$Asym[5])*exp(-exp(df$scal[5])*x)
y2  <- df$Asym[4]+(df$xmid[4]-df$Asym[4])*exp(-exp(df$scal[4])*x)
y12 <- c(y1,rev(y2),y1[1])  
x12 <- c(x,rev(x),x[1])  
polygon(x12,y12,border=NA,col=defblue)
dev.off()