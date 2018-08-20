#include <string>
#include <vector>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "mtcnn/mtcnn_detector.cc"

#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <stdio.h>

using std::string;
using std::vector;
using namespace cv;

#define LOG_TAG "Face_Det-JNI"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


//#ifdef __cplusplus
extern "C" {
//#endif

TuringOS::MtcnnDetector *md;
vector<TuringOS::BoundingBox> results;

///////////////////////////////////////////////////////////////////////////////////////////version 1
///////////////////////////////////////////////////////////////////////////////////////////added by luyuhao
struct VisionDetRetOffsets {
    jfieldID label;
    jfieldID confidence;
    jfieldID left;
    jfieldID top;
    jfieldID right;
    jfieldID bottom;
} gVisionDetRetOffsets;

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniNativeClassInit
(JNIEnv* _env, jclass _this)
{
    jclass detRetClass = _env->FindClass("turing/os/vision/bean/VisionDetRet");
    if(detRetClass){
        ;//LOGD("detRetClass is not null");
    }
    gVisionDetRetOffsets.confidence = _env->GetFieldID(detRetClass, "mConfidence", "F");
    gVisionDetRetOffsets.left = _env->GetFieldID(detRetClass, "mLeft", "I");
    gVisionDetRetOffsets.top = _env->GetFieldID(detRetClass, "mTop", "I");
    gVisionDetRetOffsets.right = _env->GetFieldID(detRetClass, "mRight", "I");
    gVisionDetRetOffsets.bottom = _env->GetFieldID(detRetClass, "mBottom", "I");
    return 1;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniCreatePredictor
(JNIEnv *env, jclass _this, jint dev_type, jint dev_id, jint minsize, jfloatArray jmeta_params)
{
    jfloat* meta_params = env->GetFloatArrayElements(jmeta_params, 0);
    vector<float> threshold;
    float factor = meta_params[0];
    for (int i=1;i<5;i++){
        threshold.push_back(meta_params[i]);
    }
    if(!md){
        md = new TuringOS::MtcnnDetector(minsize, threshold, factor, dev_type, dev_id);
    }

    env->ReleaseFloatArrayElements(jmeta_params, meta_params, 0);
    return 1;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniFree
(JNIEnv *, jclass)
{
    if(md != NULL){
        delete md;
        md = NULL;
    }
    return 1;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniFaceDetect
(JNIEnv *env, jclass, jbyteArray data1, jint width, jint height)
{
    jbyte* data2 = env->GetByteArrayElements(data1, NULL);  //jbyteArray->uchar*
    Mat frameRGBA = Mat(height,width, CV_8UC4, (uchar*)data2);  // byte array to Mat RGBA
    env->ReleaseByteArrayElements(data1, data2, JNI_ABORT);

    Mat frameBGR;
    cvtColor(frameRGBA, frameBGR, COLOR_RGBA2BGR);

    results = md->detect_face(frameBGR);
    int numOfFaceDetected = results.size();
    return numOfFaceDetected;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniGetRect
(JNIEnv* env, jclass, jobject detRet, jint index)
{
    if(md) {
        TuringOS::BoundingBox bb = results[index];
        int left = bb.x1;
        int right = bb.x2;
        int top = bb.y1;
        int bottom = bb.y2;

        env->SetIntField(detRet, gVisionDetRetOffsets.left, left);
        env->SetIntField(detRet, gVisionDetRetOffsets.top, top);
        env->SetIntField(detRet, gVisionDetRetOffsets.right, right);
        env->SetIntField(detRet, gVisionDetRetOffsets.bottom, bottom);
        return 1;
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////version 2
///////////////////////////////////////////////////////////////////////////////////////////added by zoucheng, 20170509
struct VisionDetRetOffsets2 {
    jfieldID label;
    jfieldID confidence;
    //location on original image
    jfieldID left;
    jfieldID top;
    jfieldID right;
    jfieldID bottom;
    //location on transformed image
    jfieldID x_t;
    jfieldID y_t;
    jfieldID w_t;
    jfieldID h_t;
    //rotate status of transformed image
    jfieldID s_t;
} gVisionDetRetOffsets2;

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniNativeClassInit2
(JNIEnv* _env, jclass _this)
{
    jclass detRetClass = _env->FindClass("turing/os/vision/bean/VisionDetRet");
    if(detRetClass){
        ;//LOGD("detRetClass is not null");
    }
    gVisionDetRetOffsets2.confidence = _env->GetFieldID(detRetClass, "mConfidence", "F");
    gVisionDetRetOffsets2.left = _env->GetFieldID(detRetClass, "mLeft", "I");
    gVisionDetRetOffsets2.top = _env->GetFieldID(detRetClass, "mTop", "I");
    gVisionDetRetOffsets2.right = _env->GetFieldID(detRetClass, "mRight", "I");
    gVisionDetRetOffsets2.bottom = _env->GetFieldID(detRetClass, "mBottom", "I");

    gVisionDetRetOffsets2.x_t = _env->GetFieldID(detRetClass, "mTransformedX", "I");
    gVisionDetRetOffsets2.y_t = _env->GetFieldID(detRetClass, "mTransformedY", "I");
    gVisionDetRetOffsets2.w_t = _env->GetFieldID(detRetClass, "mTransformedW", "I");
    gVisionDetRetOffsets2.h_t = _env->GetFieldID(detRetClass, "mTransformedH", "I");
    gVisionDetRetOffsets2.s_t = _env->GetFieldID(detRetClass, "mTransformedS", "I");
    return 1;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniCreatePredictor2
(JNIEnv *env, jclass _this, jint dev_type, jint dev_id, jint minsize, jfloatArray jmeta_params)
{
    jfloat* meta_params = env->GetFloatArrayElements(jmeta_params, 0);
    vector<float> threshold;
    float factor = meta_params[0];
    for (int i=1;i<5;i++){
        threshold.push_back(meta_params[i]);
    }
    if(!md){
        md = new TuringOS::MtcnnDetector(minsize, threshold, factor, dev_type, dev_id);
    }

    env->ReleaseFloatArrayElements(jmeta_params, meta_params, 0);
    return 1;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniFree2
(JNIEnv *, jclass)
{
    if(md != NULL){
        delete md;
        md = NULL;
    }
    return 1;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniFaceDetect2
(JNIEnv *env, jclass, jbyteArray data1, jint width, jint height)
{
    jbyte* data2 = env->GetByteArrayElements(data1, NULL);  //jbyteArray->uchar*
    Mat frameRGBA = Mat(height,width, CV_8UC4, (uchar*)data2);  // byte array to Mat RGBA
    env->ReleaseByteArrayElements(data1, data2, JNI_ABORT);

    Mat frameBGR;
    cvtColor(frameRGBA, frameBGR, COLOR_RGBA2BGR);

    results = md->detect_face2(frameBGR, true);
    int numOfFaceDetected = results.size();
    return numOfFaceDetected;
}

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_MtcnnPredictor_jniGetRect2
(JNIEnv* env, jclass, jobject detRet, jint index)
{
    if(md) {
        TuringOS::BoundingBox bb = results[index];
        int left = bb.x1;
        int right = bb.x2;
        int top = bb.y1;
        int bottom = bb.y2;

        cv::Rect rect_transformed = bb.rect_transformed;
        int x_t = rect_transformed.x;
        int y_t = rect_transformed.y;
        int w_t = rect_transformed.width;
        int h_t = rect_transformed.height;
        int s_t = bb.status;

        env->SetIntField(detRet, gVisionDetRetOffsets2.left, left);
        env->SetIntField(detRet, gVisionDetRetOffsets2.top, top);
        env->SetIntField(detRet, gVisionDetRetOffsets2.right, right);
        env->SetIntField(detRet, gVisionDetRetOffsets2.bottom, bottom);
        
        env->SetIntField(detRet, gVisionDetRetOffsets2.x_t, x_t);
        env->SetIntField(detRet, gVisionDetRetOffsets2.y_t, y_t);
        env->SetIntField(detRet, gVisionDetRetOffsets2.w_t, w_t);
        env->SetIntField(detRet, gVisionDetRetOffsets2.h_t, h_t);
        env->SetIntField(detRet, gVisionDetRetOffsets2.s_t, s_t);
        return 1;
    }
    return 0;
}

//#ifdef __cplusplus
}
//#endif


