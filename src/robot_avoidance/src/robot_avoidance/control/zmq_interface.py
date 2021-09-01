import traceback

import zmq


class ZMQInterface(object):
    """
        ZMQ Interface class for communication via ZMQ.
        Available methods (for usage, see documentation at function definition):
        - add_subscriber
        - add_publisher
        - send
        - receive
        - poll_command
    """

    def __init__(self):
        """
        Constructor of the ZMQInterface class.
        """
        self.context = zmq.Context()
        self.subscribers = {}
        self.publishers = {}

    @staticmethod
    def is_connected():
        # TODO python bindings don't provide 'is_connected'...
        return True

    def add_subscriber(self, subscriber_uri):
        """
        Add and initialize subscriber with a desired URI.
        :param subscriber_uri: URI of the publisher that should send the message
        :type subscriber_uri: str
        """
        assert isinstance(subscriber_uri,
                          str), "[ZMQInterface::add_subscriber] Argument 'subscriber_uri' of wrong type."
        if subscriber_uri not in self.subscribers.keys():
            subscriber = self.context.socket(zmq.SUB)
            subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            subscriber.setsockopt(zmq.CONFLATE, 1)
            subscriber.bind("tcp://" + subscriber_uri)
            self.subscribers[subscriber_uri] = subscriber
            return True
        else:
            raise ValueError(
                "[ZMQInterface::add_subscriber] There has already been a subscriber registered to this URI!")

    def add_publisher(self, publisher_uri):
        """
        Add and initialize publisher with a desired URI.
        :param publisher_uri: URI of the publisher that should send the message
        :type publisher_uri: str
        """
        assert isinstance(publisher_uri, str), "[ZMQInterface::add_publisher] Argument 'publisher_uri' of wrong type."
        if publisher_uri not in self.publishers.keys():
            publisher = self.context.socket(zmq.PUB)
            publisher.bind("tcp://" + publisher_uri)
            self.publishers[publisher_uri] = publisher
        else:
            raise ValueError("[ZMQInterface::add_publisher] There has already been a publisher registered to this URI!")

    def send(self, publisher_uri, message):
        """
        Send message from ZMQ publisher.
        :param publisher_uri: URI of the publisher that should send the message
        :param message:
        :type publisher_uri: str
        :type message: bytes
        :return: Boolean if sending was successful
        :rtype: bool
        """
        assert isinstance(publisher_uri, str), "[ZMQInterface::send] Argument 'publisher_uri' of wrong type."
        assert isinstance(message, bytes), "[ZMQInterface::send] Argument 'message' of wrong type."
        if publisher_uri in self.publishers.keys():
            res = self.publishers[publisher_uri].send(message, flags=0)
            return res is None
        else:
            raise ValueError("[ZMQInterface::send] There has not been a publisher registered at this URI!")

    def receive(self, subscriber_uri, flags=0):
        """
        Receive message from ZMQ socket.$
        :param subscriber_uri: URI of the subscriber that should receive a message
        :param flags: ZMQ flags
        :type subscriber_uri: str
        :type flags: int
        :return: The message content if a message was received, False if there was no new message received
        """
        assert isinstance(subscriber_uri, str), "[ZMQInterface::receive] Argument 'subscriber_uri' of wrong type."
        assert isinstance(flags, int), "[ZMQInterface::receive] Argument 'flags' of wrong type."
        if subscriber_uri in self.subscribers.keys():
            try:
                message = self.subscribers[subscriber_uri].recv(flags=flags)
                return message
            except zmq.ZMQError as e:
                if e.errno is not zmq.EAGAIN:
                    traceback.print_exc()
                return False
        else:
            raise ValueError("[ZMQInterface::receive] There has not been a subscriber registered at this URI!")

    def poll(self, subscriber_uri):
        """
        Receive message from ZMQ socket with flag NOBLOCK.
        :param subscriber_uri: URI of the subscriber that should receive a message
        :type subscriber_uri: str
        :return: The message content if a message was received, False if there was no new message received
        """
        assert isinstance(subscriber_uri, str), "[ZMQInterface::poll] Argument 'subscriber_uri' of wrong type."
        return self.receive(subscriber_uri, flags=zmq.NOBLOCK)
