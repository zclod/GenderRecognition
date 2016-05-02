def my_siamese_loss(y_true, y_pred):

    v_pari = y_pred[0::2]
    v_dispari = y_pred[1::2]
    y_pari = y_true[0::2]
    y_dispari = y_true[1::2]

    # TODO: se non funziona si può rimuovere e provare senza
    # Normalizza con l2 norm
    L2_pari = T.sqrt(T.sum(v_pari ** 2))
    L2_dispari = T.sqrt(T.sum(v_dispari ** 2))
    v_pari = v_pari/L2_pari
    v_dispari = v_dispari/L2_dispari

    d = T.sqr(v_pari - v_dispari)
    l = T.sqrt(T.sum(d, axis=1))

    loss = 0.5 * (T.transpose(y_pari) * T.sqr(l) + T.transpose(1-y_pari) * T.sqr(T.maximum(margin-l,0)))
    loss = T.mean(loss)

    return loss
