#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

struct packet
{
    int packetID;
    int payloadSize;
    char* payload;
};


char* serialize(struct packet *p){
    //serialize struct
}


int main()
{
    // creating socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    // specifying address
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(32556);
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    // inet_aton("127.0.0.1", &serverAddress.sin_addr.s_addr);

    // sending connection request
    int d = connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
    cout << "d = " << d << endl;

    // sending data
    const char *message = "Hello, server!";
    send(clientSocket, message, strlen(message), 0);

    // closing socket
    close(clientSocket);

    return 0;
}


