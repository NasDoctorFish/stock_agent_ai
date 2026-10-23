# proxy server 4000
# npx ts-node src/server.ts

#  Correct Server Startup                                                        
                                                                                
  # Port 4000 (proxy + OTP + JWT)                                               
#  npm run dev          # or: npm start                                          
                                                                                
  # Port 3000 (order server)                                                    
#  npm run start:order        


pm2 start npm --name "CoinOne_Proxy" -- run dev
pm2 start npm --name "CoinOne_Order" -- run start:order                                           
                      
# pm2 show "CoinOne_Proxy"
# pm2 info "CoinOne_Proxy"
# pm2 logs CoinOne_Proxy
# pm2 list

#Auto restartss
pm2 save
pm2 startup