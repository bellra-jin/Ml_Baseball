# TPOT Best Pipeline (auto-generated)
# Generated: 2026-04-29T12:15:33.196264
# Pipeline string:
# Pipeline(memory='auto',
         steps=[('normalizer', Normalizer(norm=np.str_('l2'))),
                ('selectpercentile',
                 SelectPercentile(percentile=87.4306730968204)),
                ('featureunion-1',
                 FeatureUnion(transformer_list=[('featureunion',
                                                 FeatureUnion(transformer_list=[('zerocount',
                                                                                 ZeroCount())])),
                                                ('passthrough',
                                                 Passthrough())])),
                ('featureunion-2',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('logisticregression',
                 LogisticRegression(C=1330.4329413670953,
                                    class_weight='balanced', max_iter=1000,
                                    n_jobs=1, penalty=np.str_('l1'),
                                    random_state=42, solver='saga'))])

import numpy as np
import pandas as pd

# Exported pipeline
exported_pipeline = Pipeline(memory='auto',
         steps=[('normalizer', Normalizer(norm=np.str_('l2'))),
                ('selectpercentile',
                 SelectPercentile(percentile=87.4306730968204)),
                ('featureunion-1',
                 FeatureUnion(transformer_list=[('featureunion',
                                                 FeatureUnion(transformer_list=[('zerocount',
                                                                                 ZeroCount())])),
                                                ('passthrough',
                                                 Passthrough())])),
                ('featureunion-2',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('logisticregression',
                 LogisticRegression(C=1330.4329413670953,
                                    class_weight='balanced', max_iter=1000,
                                    n_jobs=1, penalty=np.str_('l1'),
                                    random_state=42, solver='saga'))])
