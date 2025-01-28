# Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry (Apache License 2.0)
# Adjusted by Michael Plattner for lecture "Automated Driving"

class UDPCommunication:
    """
    UDP communication has to use a custom protocol:
            Byte 0:   packet number
            Byte 1:   total number of packets for this message
            Byte 2-n: data
    UDP packets need to arrive in the correct order
    """

    def __init__(self, udp_ip, port_tx, port_rx, stop_thread, enable_rx=False, suppress_warnings=True):
        """
        Constructor
        :param udp_ip: Must be string e.g. "127.0.0.1"
        :param port_tx: integer number e.g. 8000. Port to transmit from i.e From Python to other application
        :param port_rx: integer number e.g. 8001. Port to receive on i.e. From other application to Python
        :param enable_rx: When False you may only send from Python and not receive. If set to True a thread is created
                          to enable receiving of data
        :param suppress_warnings: Stop printing warnings if not connected to other application
        """

        import socket

        self.udp_ip = udp_ip
        self.port_tx = port_tx
        self.port_rx = port_rx
        self.enable_rx = enable_rx
        self.suppress_warnings = suppress_warnings  # when true warnings are suppressed
        self.rx_data = None
        self.rx_buffer = ""
        self.exp_packet_number = 1

        # Connect via UDP
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # internet protocol, udp (DGRAM) socket
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allows the address/port to be reused immediately instead of it being stuck in the TIME_WAIT state waiting for late packets to arrive.
        self.udp_sock.bind((udp_ip, port_rx))

        # Create Receiving thread if required
        if enable_rx:
            import threading
            self.rx_thread = threading.Thread(target=self.read_udp_thread_func, args=(stop_thread, ), daemon=True)
            self.rx_thread.start()

    def __del__(self):
        self.close_socket()

    def close_socket(self):
        """
        Function to close socket
        """
        self.udp_sock.close()

    def send_data(self, str_to_send):
        """
        Use this function to send string to C#
        :param str_to_send: string to send
        """
        self.udp_sock.sendto(bytes(str_to_send, 'utf-8'), (self.udp_ip, self.port_tx))

    def receive_data(self):
        """
        Should not be called by user
        Function BLOCKS until data is returned from C#. It then attempts to convert it to string and returns
        on successful conversion.
        A warning/an error is raised if:
            - Warning: Not connected to C# application yet. Warning can be suppressed by setting
                       suppress_warning=True in constructor
            - Error: If data receiving procedure or conversion to string goes wrong
            - Error: If user attempts to use this without enabling RX
        :return: returns None on failure or the received string on success
        """
        if not self.enable_rx:  # if RX is not enabled, raise error
            raise ValueError(
                "Attempting to receive data without enabling this setting. Ensure this is enabled from the constructor")

        data = None
        try:
            data, _ = self.udp_sock.recvfrom(100000)
            data = data.decode('utf-8')
        except WindowsError as e:
            if e.winerror == 10054:  # An error occurs if you try to receive before connecting to other application
                if not self.suppress_warnings:
                    print("Are You connected to the other application? Connect to it!")
                else:
                    pass
            else:
                raise ValueError("Unexpected Error. Are you sure that the received data can be converted to a string")

        return data

    def read_udp_thread_func(self, stop_thread):  # Should be called from thread
        """
        This function should be called from a thread [Done automatically via constructor]
           (import threading -> e.g. rx_thread = threading.Thread(target=self.read_udp_thread_func, daemon=True))
        This function keeps looping through the BLOCKING receive_data function. It stores the data according to the
           custom protocal in self.rx_buffer. When all packets of a message arrived, self.rx_data is set and can be
           later read by the user.
        """

        while True:
            if not stop_thread():
                self.close_socket()
                break

            packet = self.receive_data()  # Blocks until data is returned (OR MAYBE UNTIL SOME TIMEOUT AS WELL)

            if packet is not None:
                packet_number = ord(packet[0])
                if packet_number == self.exp_packet_number:
                    self.rx_buffer += packet[2:]
                    self.exp_packet_number = self.exp_packet_number + 1

                    if packet[0] == packet[1]:
                        self.rx_data = self.rx_buffer
                        self.rx_buffer = ""
                        self.exp_packet_number = 1
                else:
                    self.rx_buffer = ""

    def read_received_data(self):
        """
        This is the function that should be used to read received data. It is not possible to re-read the data.
        :return: data or None if nothing has been received
        """
        data = self.rx_data
        self.rx_data = None
        return data
