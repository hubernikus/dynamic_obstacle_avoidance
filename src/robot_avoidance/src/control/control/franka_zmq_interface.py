import struct
import time

from .zmq_interface import ZMQInterface


class FrankaZMQInterface(object):

    def __init__(self, state_uri="0.0.0.0:1601", command_uri="0.0.0.0:1602"):
        """
        Constructor of the FrankaZMQInterface class, binds and publishes to the ZMQ sockets defined by
        their URIs.
        :param state_uri: URI of the socket over which the robot state is transmitted
        :param command_uri: URI of the socket over which the command is transmitted

        :type state_uri: str
        :type command_uri: str
        """
        # set up ZMQ interface
        self.state_uri = state_uri
        self.command_uri = command_uri
        self.zmq_interface = ZMQInterface()
        self.zmq_interface.add_publisher(self.command_uri)
        self.zmq_interface.add_subscriber(self.state_uri)

        self.data_types = {'d': 8}

    def is_connected(self):
        return self.zmq_interface.is_connected()

    def get_robot_state(self):
        """
        Publish robot state to ZMQ socket.
        :param state: Current robot state as defined in bullet_robot.py
        :type state: BulletRobotState
        :return: Boolean if sending was successful
        :rtype: bool
        """
        message = self.zmq_interface.poll(self.state_uri)
        return self._decode_message(message, 'd') if message else None

    def send_command(self, command):
        """
        Receive message from ZMQ socket and decode it into a command message. If no new messages have been received
        over a defined time horizon, a timeout is triggered. The command is available at self.current_command and the
        timeout flag at self.timeout_triggered.
        :return: Current command
        :rtype: list of float
        """
        encoded_state = self._encode_message(command, 'd')
        return self.zmq_interface.send(self.command_uri, encoded_state)

    def _decode_message(self, message, data_type):
        """
        Decode message from ZMQ socket.
        :param message: Message from ZMQ socket
        :param data_type: Data type of the message
        :type message: bytes
        :type data_type: str
        :return: Decoded message
        :rtype: Any
        """
        return [struct.unpack(data_type, message[i:i + self.data_types[data_type]])[0] for i in
                range(0, len(message), self.data_types[data_type])]

    @staticmethod
    def _encode_message(message, data_type):
        """
        Encode message to send it over ZMQ socket.
        :param message: list of floats
        :param data_type: data type of the message
        :type message: Any
        :type data_type: str
        :return: State as list of bytes
        :rtype: bytes
        """
        return b"".join([struct.pack(data_type, message[i]) for i in range(len(message))])
