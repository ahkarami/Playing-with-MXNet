# Amir Hossein Karami
# Subject: Save & Load Model with MXNet Deep Learning Framework


# ****** Address Save Model in New MXNet version:
# # save a model to mymodel-symbol.json and mymodel-0100.params
# face_model = get_face_representation_model(face_model_address)
# prefix = 'ResNet50'
# iteration = 0000
# # face_model.save(prefix, iteration)
# # net.save('xxx.params')
#
# epoch = 0
# sym, arg_params, aux_params = mx.model.load_checkpoint(face_model_address + 'model', epoch)
# mx.model.save_checkpoint(prefix, iteration, sym, arg_params, aux_params)
#
# # sym.save(prefix, iteration)


# 1- https://mxnet.incubator.apache.org/api/python/model.html#save-the-model
# 2- https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html
# 3- https://discuss.mxnet.io/t/save-cnn-model-architecture-and-params/683/9
# 4- https://discuss.mxnet.io/t/efficient-way-of-saving-gluon-trainer/357/2
# 5- https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.Symbol.save

