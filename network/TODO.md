# Roadmap of TODOs

 1. adding new nodes to the network (controlled by the verifier nodes)
   1. [ ] 1.1. add a new node to the network should be performed either by the network administrator or by the node itself (in case of malicious nodes)
  2. [ ] 1.2. the creator of the new node will send to the node the verifier node list 
  3. [ ] 1.3. the new node will send the its first message to the verifier nodes with the following information:
     1. [ ] 1.3.1. the new node public key
  2. [ ] 1.4. the verifier nodes will register the new node in the network and say to him in which round he will be able to participate and its score(the current one)
  3. [ ] 1.5. the new node will receive the current round information and will be able to participate in the network

    THIS RESOLVE THE ISSUE #1 (adding new nodes to the network)


  2. finish the ROUND (controlled by the verifier nodes)
  2. [ ] 2.1. We eliminate the not participating nodes, the timeout problem and failures by expecting no failure in the network
  3. [ ] 2.2. The verifier nodes will send "start round" message to the training nodes with the score of all nodes
  4. [ ] 2.3. The training nodes will start the training process and send the message with all the information to the verifier nodes
  5. [ ] 2.4. The verifier nodes will receive the information 
  6. [ ] 2.5. When the verifier nodes receive all the information, they will compute the block and send it to the network