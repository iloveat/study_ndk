# Change this to path of yours
export MXNET_ROOT = third/mxnet
export NNPACK_ROOT = third/nnpack
export OPENBLAS_ROOT = third/openblas
export CURL_ROOT = third/curl
export OPENSSL_ROOT = third/openssl
export OPENCV_ROOT = third/opencv
export ARCHITECTURE = armeabi-v7a-hard


export CXX=arm-linux-androideabi-clang++

#commonly used:
CFLAGS = -std=c++11 -Wno-unknown-pragmas -Wall
CFLAGS += -I${OPENBLAS_ROOT} -I${OPENBLAS_ROOT}/include
LDFLAGS += -L${OPENBLAS_ROOT} -L${OPENBLAS_ROOT}/lib -lopenblas
#mxnet itself:
CFLAGS += -I${MXNET_ROOT}
CFLAGS += -I${MXNET_ROOT}/dmlc-core/include
CFLAGS += -I${MXNET_ROOT}/include
CFLAGS += -I${MXNET_ROOT}/mshadow
#nnpack:
CFLAGS += -DMXNET_USE_NNPACK=1
CFLAGS += -DMXNET_USE_NNPACK_NUM_THREADS=4
CFLAGS += -I${NNPACK_ROOT}/include
LDFLAGS += -L${NNPACK_ROOT}/obj/local/armeabi-v7a
LDFLAGS += -lnnpack -lpthreadpool -lnnpack_ukernels -lcpufeatures
#nnpack dependence googletest:
CFLAGS += -I${NNPACK_ROOT}/third-party/gtest-1.7.0/include
CFLAGS += -I${NNPACK_ROOT}/third-party/gtest-1.7.0
#nnpack dependence pthreadpool:
CFLAGS += -I${NNPACK_ROOT}/third-party/pthreadpool/include
CFLAGS += -I${NNPACK_ROOT}/third-party/FXdiv/include
#other define used
CFLAGS += -DMSHADOW_STAND_ALONE=1
CFLAGS += -DMSHADOW_USE_CUDA=0
CFLAGS += -DMSHADOW_USE_MKL=0
CFLAGS += -DSHADOW_RABIT_PS=0
CFLAGS += -DMSHADOW_DIST_PS=0
CFLAGS += -DMSHADOW_USE_SSE=0
CFLAGS += -DMXNET_USE_OPENCV=0
CFLAGS += -DMXNET_PREDICT_ONLY=0
CFLAGS += -DDISABLE_OPENMP=1
#curl dependencies:
CFLAGS += -I$(CURL_ROOT)/include
LDFLAGS += $(CURL_ROOT)/lib/libcurl.a -lssl -lcrypto
#openssl dependencies
CFLAGS += -I$(OPENSSL_ROOT)/include
LDFLAGS += $(OPENSSL_ROOT)/lib/libcrypto.a
LDFLAGS += $(OPENSSL_ROOT)/lib/libssl.a
#user defined libs:
CFLAGS += -Ijni/validation
LDFLAGS += libvalidation.so


all: encryption mtcnn


# predictor 
CFLAGS += -D_NDK_MATH_NO_SOFTFP=1 -O3 -mhard-float
LDFLAGS += -Wl,--no-warn-mismatch -lm_hard
LDFLAGS += -llog -ldl -lz

libmxnet_predict.o: jni/mxnet/mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c jni/predictor.cc

libmxnet_predict.so: libmxnet_predict.o
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

android: libmxnet_predict.so


# encryption
libmxnet_predict_encrypt.o: jni/mxnet/mxnet_predict-all.cc 
	${CXX} ${CFLAGS} -fPIC -o $@ -c jni/predictor_encrypt.cc

libmxnet_predict_encrypt.so: libmxnet_predict_encrypt.o
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

encryption: libmxnet_predict_encrypt.so


# android mtcnn
CFLAGS_MTCNN = $(CFLAGS) -std=c++11 -I$(OPENCV_ROOT)/jni/include -I$(OPENCV_ROOT)/jni/include/opencv -I$(MXNET_ROOT)/amalgamation/jni/mtcnn
CFLAGS_MTCNN += -D_NDK_MATH_NO_SOFTFP=1 -O3 -mhard-float
LDFLAGS_MTCNN = $(LDFLAGS) -L$(OPENCV_ROOT)/libs/$(ARCHITECTURE)
LDFLAGS_MTCNN += -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab -lopencv_ts
LDFLAGS_MTCNN += -lopencv_java3
LDFLAGS_MTCNN += $(OPENCV_ROOT)/3rdparty/libs/armeabi-v7a-hard/libIlmImf.a
LDFLAGS_MTCNN += $(OPENCV_ROOT)/3rdparty/libs/armeabi-v7a-hard/liblibjasper.a
LDFLAGS_MTCNN += $(OPENCV_ROOT)/3rdparty/libs/armeabi-v7a-hard/liblibjpeg.a
LDFLAGS_MTCNN += $(OPENCV_ROOT)/3rdparty/libs/armeabi-v7a-hard/liblibpng.a
LDFLAGS_MTCNN += $(OPENCV_ROOT)/3rdparty/libs/armeabi-v7a-hard/liblibtiff.a
LDFLAGS_MTCNN += $(OPENCV_ROOT)/3rdparty/libs/armeabi-v7a-hard/liblibwebp.a
LDFLAGS_MTCNN += -Wl,--no-warn-mismatch -lm_hard

libmtcnn_facedetector.o: jni/mxnet/mxnet_predict-all.cc
	${CXX} ${CFLAGS_MTCNN} -fPIC -o $@ -c jni/mtcnn_predictor.cc

libmtcnn_facedetector.so: libmtcnn_facedetector.o
	${CXX} ${CFLAGS_MTCNN} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS_MTCNN)

mtcnn: libmtcnn_facedetector.so



.PHONY: all clean

clean:
	rm -f *.o libmxnet_predict.so libmxnet_predict_encrypt.so libmtcnn_facedetector.so







