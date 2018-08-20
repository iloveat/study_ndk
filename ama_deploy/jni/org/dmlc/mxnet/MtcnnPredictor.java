package org.dmlc.mxnet;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import turing.os.vision.bean.VisionDetRet;

/**
 * Created by root on 17-2-22.
 */
public class MtcnnPredictor {
    private static final String TAG = "mtcnn_facedetector";
    protected static boolean sInitialized = false;
    private int detId=0;

    static {
        try{
            System.loadLibrary("mtcnn_facedetector");
            android.util.Log.d("mtcnn_facedetector", "load mtcnn_facedetector success");
        } catch (UnsatisfiedLinkError e) {
            android.util.Log.d("mtcnn_facedetector", "library not found!");
            Log.e(TAG, "library not found! error ", e);
        }
    }

    public void init(){
        //String modelPath = "/sdcard/turingos/model/";
        int devType = 1;
        int devId = 0;
        int minSize = 50;
        float factor1 = 0.44f;// image scale factor
        float thres1 = 0.9f;//1st nn threshold 1
        float thres2 = 0.9f;//1st nn threshold 2
        float thres3 = 0.7f;//2st nn threshold
        float thres4 = 0.7f;//3st nn threshold
        float[] metaParams = {factor1, thres1, thres2, thres3, thres4};
        jniNativeClassInit();
        jniCreatePredictor(devType, devId, minSize, metaParams);
    }

    public void free(){
        jniFree();
    }
    /**
     * this function convert argb_8888 bitmap to size w*h*4 byteArray
     */
    private byte[] bitmap2byte(final Bitmap bitmap){
        ByteBuffer byteBuffer = ByteBuffer.allocate(bitmap.getByteCount());
        bitmap.copyPixelsToBuffer(byteBuffer);
//		Log.d(TAG, "asdf1, detect image config " + bitmap.getConfig().toString());
//		Log.d(TAG, "asdf1, detect image size: " + bitmap.getByteCount());
        byte[] bytes = byteBuffer.array();
        return bytes;
    }

    public List<VisionDetRet> detect(final Bitmap bitmap) {
        long sss = System.currentTimeMillis();
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();

        long startJNI = SystemClock.uptimeMillis();
        byte[] bytes = bitmap2byte(bitmap);
        Log.d("FaceLog", "bitmap2byte time: " + (SystemClock.uptimeMillis() - startJNI));

        List<VisionDetRet> ret = new ArrayList<VisionDetRet>();
        int size = jniFaceDetect(bytes, w, h); // 80 for mode 1
        Log.d(TAG, "Face_Det time: " + (System.currentTimeMillis() - sss));
        //int size = jniDLibHOGDetect(path);
        //Log.d(TAG, "-----size--- : " + size);
        for (int i = 0; i != size; i++) {
            long start = System.currentTimeMillis();
            VisionDetRet det = new VisionDetRet();
            det.setEmotion("");
            int success1 = jniGetRect(det, i);
            //Log.d(TAG, "jniGetDLibRet done : "+success1);
            if (success1 >= 0) {
                //if (success1>=0){
                detId++;if (detId==100) detId=0;
                det.setLabel(Integer.toString(detId));
                ret.add(det);
            }
        }
        return ret;
    }

    private native static int jniNativeClassInit();
    private native static int jniCreatePredictor(int devType, int devId, int minSize, float[] metaParams);
    private native static int jniFree();
    private native static int jniFaceDetect(byte[] image, int width, int height);
    private native static int jniGetRect(VisionDetRet det, int index);

}
