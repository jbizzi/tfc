import numpy as np

# Generator matrix G
G = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 1, 1, 1]])

# Parity-check matrix H
H = np.array([[0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 1]])


# Encode function
def encode(message):
    # Ensure message length is 4
    assert len(message) == 4, "Message length must be 4 bits"
    message_array = np.array([int(bit) for bit in message])
    # Reshape message array to a column vector
    message_array = np.reshape(message_array, (4, 1))
    codeword = np.dot(G, message_array) % 2
    return ''.join(map(str, np.reshape(codeword, (7,))))


# Decode function
def decode(received):
    received_array = np.array([int(bit) for bit in received])
    received_array = np.reshape(received_array, (7, 1))
    syndrome = np.dot(H, received_array) % 2
    error_position = np.sum(syndrome * [1, 2, 4])
    if error_position != 0:
        # Flip the bit at the error position
        received_array[error_position - 1] = 1 - received_array[error_position - 1]
    # Extract original message
    original_message = received_array[[2, 4, 5, 6]]  # Corrected indices to be 0-based
    return ''.join(map(str, np.reshape(original_message, (4,))))


# Example usage
message = "1011"
print("Original message:", message)
encoded_message = encode(message)
print("Encoded message:", encoded_message)
# Simulate a transmission error
received_message = encoded_message
# Flip one bit
received_message = received_message[:3] + str(1 - int(received_message[3])) + received_message[4:]
print("Received message with error:", received_message)
decoded_message = decode(received_message)
print("Decoded message:", decoded_message)
