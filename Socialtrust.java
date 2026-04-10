/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trust_system_lib;
/**
 *
 * @author Fatimah almuzaini
 */

import core_lib.*;
import java.util.*;

public class Socialtrust implements TrustAlg {
        private Trustinfo LL [] [];
  private globalTrust TL[];
    private globalTrust UL[];
    private Network nw;
    private int[][] friends_array;
        private int[][] acq_array;
    private int[][] size;
      private int[][] asize;
     private int NUM_USERS;
     private int random_friend;
     private int user_friend;
         private int p;
     private double pt;
       Random rand = new Random();
       boolean friend_flag;
       boolean acq_flag;
       boolean fof_flag;
      // boolean aoa_flag;
  public Socialtrust(Network nw){
       this.nw = nw;
        NUM_USERS = this.nw.GLOBALS.NUM_USERS;
        LL=new Trustinfo [NUM_USERS][NUM_USERS];
        TL=new globalTrust [NUM_USERS];
        UL=new globalTrust [NUM_USERS];
        size = new int[NUM_USERS][1];
          asize = new int[NUM_USERS][1];
        friends_array = new int[NUM_USERS][NUM_USERS];
          acq_array = new int[NUM_USERS][NUM_USERS];
          for(int i = 0 ; i < NUM_USERS; i++){// initializing  gloable lists
              UL[i]= new globalTrust();
              TL[i]=new globalTrust(false);
              asize[i][0]=0;
                for(int j = 0 ; j < NUM_USERS; j++){ 
                    friends_array[i][j] = -1;
                    LL[i][j]= new Trustinfo();
                    
                }
                //initializing  friends for local list
                pt=NUM_USERS*0.2;
                p=(int)pt;
               
                
                    user_friend= rand.nextInt((p)) ;
                     user_friend= user_friend+p;
                          size[i][0] = user_friend;
                          
                    for(int x=0;x<user_friend;x++){
                        random_friend=rand.nextInt((NUM_USERS));
                             while(random_friend == i || isfriend(i,random_friend)||(this.nw.getUser(random_friend).isgood()==false))
                             random_friend=rand.nextInt((NUM_USERS));
                         friends_array[i][x] = random_friend;
                        LL[i][x].addpos();
                        UL[random_friend]=new globalTrust();
                        UL[random_friend].increaseu();
                        UL[random_friend].addwp();
              
                        
                    }
                
  }
    
  }

    
    @Override
     public String fileExtension() {
        return "socialtrust";
    }


    @Override
    public String algName() {
        return "socialtrust";
    }


    @Override
public void update(Transaction trans){  
        int new_r = trans.getRecv();
	int new_s = trans.getSend();
        double t,wt,p,n;
        int nu;
  if(isfriend(new_r,new_s)==false){
		if(isacqua(new_r,new_s)==false){
                   acq_array[new_r][asize[new_r][0]] = new_s;
                asize[new_r][0]++;
                if (trans.getValid())
                    LL[new_r][asize[new_r][0]].addpos();
                else
                    LL[new_r][asize[new_r][0]].addneg(); 
                if(TL[new_s].state)
                    TL[new_s].increaseu();
                else
                 UL[new_s].increaseu();
            }else{ for(int i = 0;i<asize[new_r][0];i++){
                if (new_s == acq_array[new_r][i]){
                      if (trans.getValid()){
                      LL[new_r][i].addpos();
                      break;}
                      else{
                         LL[new_r][i].addneg();
                      break;     
                              }
                      
                }
            }} }else{
            
         for(int i = 0;i<size[new_r][0];i++){
                if (new_s == friends_array[new_r][i]){
                   if (trans.getValid()){
                      LL[new_r][i].addpos();
                      break;}
                      else{
                         LL[new_r][i].addneg();
                      break;     
                              }
                      
                }
            }
  }
      
                    
			
     
    if (trans.getValid()){
             if( UL[new_s].state){
                      UL[new_s].addwp();
                        p=UL[new_s].getwp();
                         n=UL[new_s].getwn();
                          nu=UL[new_s].getnumu();
                       t= computet(p,n,nu);
                       if(t>=0.5){
                           TL[new_s].addele(p, n, nu);
                           UL[new_s].delete();
                       }
                           
            }
                  if( TL[new_s].state){
                      TL[new_s].addwp();
                         p=TL[new_s].getwp();
                         n=TL[new_s].getwn();
                          nu=TL[new_s].getnumu();
                       t= computet(p,n,nu);
                       if(t<0.5){
                        UL[new_s].addele(p, n, nu);
                           TL[new_s].delete();
                       }
                  }
       
         
                  
                      
                   }else{
        if( UL[new_s].state){
                      UL[new_s].addwn();
                        p=UL[new_s].getwp();
                         n=UL[new_s].getwn();
                          nu=UL[new_s].getnumu();
                       t= computet(p,n,nu);
                      if(t>=0.5){
                           TL[new_s].addele(p, n, nu);
                           UL[new_s].delete();
                       }
                           
             }
                  if( TL[new_s].state){
                      TL[new_s].addwn();
                         p=TL[new_s].getwp();
                         n=TL[new_s].getwn();
                          nu=TL[new_s].getnumu();
                       t= computet(p,n,nu);
                       if(t<0.5){
                        UL[new_s].addele(p, n, nu);
                           TL[new_s].delete();
                       }
                  }
       
             
        
                  
                }
            }
        
 
 
    @Override
    public void computeTrust(int user, int cycle) {
  Opinion o1=new Opinion(0.0,0.0,1.0,1.0);
   Opinion o2=new Opinion(0.0,0.0,1.0,1.0);
      ArrayList<Integer> receiver_friends = new ArrayList<Integer>();
            ArrayList<Integer> receiver_friends_index = new ArrayList<Integer>();
            ArrayList<Integer> FOF = new ArrayList<Integer>();
            ArrayList<Integer> FOF_index = new ArrayList<Integer>();
            ArrayList<Opinion> discounts_consensuses = new ArrayList<Opinion>();
            Opinion temp_trust = new Opinion(0.0, 0.0, 1.0, 1.0);
               Opinion temp_op = new Opinion(0.0, 0.0, 1.0, 1.0);
            int temp=0;
            double pos=0.0,neg=0.0;
            double trust,maxtrust=0.0;
           for(int i=0 ;i< NUM_USERS;i++){
               trust=0.0;
           friend_flag = false;
            acq_flag = false;
            fof_flag =false;
        
               for(int j=0;j< size[user][0];j++){
               if (friends_array[user][j] == i){
            trust= computeop(LL[user][j].getpos(),LL[user][j].getneg(),1.0);  
            friend_flag=true;
            break;}}
         
               
             if(friend_flag==false){
              for(int j=0;j< asize[user][0];j++){
               if (acq_array[user][j] == i){
            trust= computeop(LL[user][j].getpos(),LL[user][j].getneg(),0.5);  
            acq_flag=true;
            break;}}}
           
               
            if (friend_flag==false && acq_flag == false){
                for(int x=0;x < size[user][0];x++)
                        for(int j = 0 ; j < size[friends_array[user][x]][0] ; j++){//receiver_fof_index
                            if(i == friends_array[friends_array[user][x]][j]){
                                if(fof_flag == false){fof_flag = true;}
                                    receiver_friends.add(friends_array[user][x]);
                                    receiver_friends_index.add(x);
                                    FOF.add(friends_array[friends_array[user][x]][j]);
                                    FOF_index.add(j);
                                
                            }
                        }
                       
                    
                
                if (!FOF.isEmpty()){
                    for(int l = 0 ; l < FOF_index.size() ; l++){
                        temp_op= getop(LL[user][receiver_friends_index.get(l)].getpos(),LL[user][receiver_friends_index.get(l)].getneg(),1.0);
                        discounts_consensuses.add(temp_op.discount(getop(LL[receiver_friends_index.get(l)][FOF_index.get(l)].getpos(),LL[receiver_friends_index.get(l)][FOF_index.get(l)].getneg(),1.0)));
                    }
            
                        for(int m = 0; m < discounts_consensuses.size() ; m++)
                            {
                                temp_trust = temp_trust.consensus(discounts_consensuses.get(m));
                            }
                        trust=temp_trust.expectedValue();
                }}
                    
                      
                    
                    
               
            
             if(friend_flag==false && acq_flag == false &&fof_flag ==false ){
                  
                       if( TL[i].state)
                      trust=computet(TL[i].getwp(),TL[i].getwn(),TL[i].getnumu());       
                   else 
                       trust=computet(UL[i].getwp(),UL[i].getwn(),UL[i].getnumu());
                 
           }
            
               
      
       Relation rel1 = this.nw.getUserRelation(user,i);
                 rel1.setTrust(trust);
           }       
         
             }
            
        
         
            private boolean isfriend (int uid,int fid){
     int s=size[uid][0];
      for(int i=0;i<s;i++){
          if(friends_array [uid][i]==fid)
              return true;}
      return false;
      
  }
             private boolean isacqua (int uid,int aquid){
     int s=asize[uid][0];
      for(int i=0;i<s;i++){
          if(acq_array [uid][i]==aquid)
              return true;}
      return false;
      
  }
             
    
             
    
    public double computet(double wp,double wn,int nu){
        
       double t ,b,a,u,v;
     
       v=variance(wp,wn);
	  
      t=(((wp*nu)/(((wp+wn)*(wp+wn))+2.0))*v);

       return t;
    }
    public double computeop(int p,int n,double a){
        double b,d,u,t;
        	b = (p / (p + n + 2.0));
		d = (n / (p + n + 2.0));
		u = (2.0 / (p + n + 2.0));
                t=(b+(a*u));
                return t;
    }
      public Opinion getop(int p,int n,double a){
        double b,d,u;
        	b = (p / (p + n + 2.0));
		d = (n / (p + n + 2.0));
		u = (2.0 / (p + n + 2.0));
                return (new Opinion(b, d, u, a));
    }
        public double variance(double pos,double neg){
        double total,p,n;
		if(pos>0.0&&neg==0.0)
			return(1.0);
		if(pos==0.0&&neg==0.0)
			return(0.0);
        total=pos+neg;
        p=pos/total;
        n=neg/total;
	
        return (p-n);
        
    }
}
