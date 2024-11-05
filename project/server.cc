// C++ program to show the example of server application in
// socket programming
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

int main()
{
     // creating socket
     int serverSocket = socket(AF_INET, SOCK_STREAM, 0);

     cout << "serverSocket = " << serverSocket << endl;

     // specifying the address
     sockaddr_in serverAddress;
     serverAddress.sin_family = AF_INET;
     serverAddress.sin_port = htons(32556);
     serverAddress.sin_addr.s_addr = INADDR_ANY;

     cout << "start binding" << endl;

     // binding socket.
     ::bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
     // ::bind(serverSocket, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress));

     cout << "start listenning" << endl;

     // listening to the assigned socket
     listen(serverSocket, SOMAXCONN);

     // accepting connection request
     sockaddr_in client;
     socklen_t clientSize = sizeof(client);
     int clientSocket = accept(serverSocket, reinterpret_cast<sockaddr *>(&client), &clientSize);

     cout << "receive from " << clientSocket << endl;

     // recieving data
     char buffer[1024] = {0};
     recv(clientSocket, buffer, sizeof(buffer), 0);
     cout << "Message from client: " << buffer
          << endl;

     // closing the socket.
     close(serverSocket);

     return 0;
}
