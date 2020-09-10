import pandas as pd


def dataframe_from_result(results):

    rows = []

    for task, config_id in enumerate(results.data):

        d = results.data[config_id]
        bracket, _, _ = config_id

        for epoch in d.results:

            row = dict(task=task,
                       bracket=bracket,
                       epoch=int(epoch),
                       loss=d.results[epoch]["loss"],
                       info=d.results[epoch]["info"],
                       submitted=d.time_stamps[epoch]["submitted"],
                       started=d.time_stamps[epoch]["started"],
                       finished=d.time_stamps[epoch]["finished"])
            row.update(d.config)
            rows.append(row)

    return pd.DataFrame(rows)
