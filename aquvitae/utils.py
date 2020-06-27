def result_to_tqdm_template(result, training=True):
    template = ""
    for k in result.keys():
        display_name = k
        if not training:
            display_name = "val_{}".format(display_name)
        template += "{}: {:.4f} -".format(display_name, result[k])
    return template[:-1]
