def result_to_tqdm_template(result):
    template = ""
    for k in result.keys():
        template += "[{}-{:.5f}] ".format(k, result[k])
    return template[:-1]
