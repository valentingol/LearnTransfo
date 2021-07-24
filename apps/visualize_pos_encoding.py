from transformer.architecture.pos_encoding import PostionalEncoding
import matplotlib.pyplot as plt

if __name__ == '__main__':
    max_denom = 1000
    seq_len = 7000
    depth = 30
    pe = PostionalEncoding(max_denom=max_denom)
    pos_code = pe(seq_len=seq_len, depth=depth)
    code_sin = pos_code[:, 0::2]
    code_cos = pos_code[:, 1::2]

    plt.subplot(211)
    plt.title('sinus encoding')
    plt.pcolormesh(code_sin.numpy().T, cmap='coolwarm')
    plt.ylabel('depths')
    plt.xlabel('positions')
    plt.colorbar()

    plt.subplot(212)
    plt.title('cosinus encoding')
    plt.pcolormesh(code_cos.numpy().T, cmap='coolwarm')
    plt.ylabel('depths')
    plt.xlabel('positions')
    plt.colorbar()

    plt.show()
