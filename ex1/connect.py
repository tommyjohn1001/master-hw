import argparse
import socket
import sys

BUFF = 1024


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", dest="mode", type=str, choices=["client", "server"], default="server"
    )
    parser.add_argument("-i", dest="host", type=str, default=None)
    parser.add_argument("-p", dest="port", type=int, default=32456)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    match args.mode:
        case "server":
            if args.host is None:
                host = socket.gethostbyname(socket.gethostname())
            else:
                host = args.host

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, args.port))
                s.listen()

                print(f"Server works at: {host} - {args.port}")

                while True:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected by {addr}")
                        while True:
                            data = conn.recv(BUFF)
                            if not data:
                                break
                            conn.sendall(data)

        case "client":
            if args.host is None:
                print("Err: In mode 'client', argument '-i' must be specified")
                return 1

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((args.host, args.port))
                s.sendall(b"Hello, world")
                data = s.recv(1024)


if __name__ == "__main__":
    sys.exit(main())

# Usage: server: python connect.py
# Usage: client: python connect.py -m client -i <server_ip>
