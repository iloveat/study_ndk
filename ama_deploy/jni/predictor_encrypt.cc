#include "org_dmlc_mxnet_PredictorEncrypt.h"
#include "aes/aes128.h"
#include "mxnet/mxnet_predict-all.cc"
#include <jni.h>
#include <android/log.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <curl/curl.h>
#include <openssl/aes.h>
#include <openssl/md5.h>
#include <openssl/hmac.h>
#include <openssl/buffer.h>
#include "validation.h"
#include "../configure/predefined.h"


#define LOG_TAG "PredictorEncrypt"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define POST_TYPE_HTTP  (0)
#define POST_TYPE_HTTPS (1)

// receive buffer size
const int RECV_BUFFER_SIZE = 1024;
// receive buffer
char recv_buff[RECV_BUFFER_SIZE];

// set CAs' location 
const std::string CA_GET_USERID = "/sdcard/turingos/myusr.crt";
const std::string CA_VERIFY_KEY = "/sdcard/turingos/mykey.pem";

// set https' request url
const std::string URL_OBTAIN_USERID = "https://www.tuling123.com/openapi/getuserid.do";
const std::string URL_VERIFY_APIKEY = "https://smartdevice.ai.tuling123.com/projectapi/multibiz";


// 通过java反射获取uuid
// 返回结果需要手动释放
char* generate_uuid(JNIEnv* env)
{
    jclass uuid_class = env->FindClass("java/util/UUID");
    if(env->ExceptionOccurred() || uuid_class == NULL)
    {
        LOGE("find UUID Failed.");
        return NULL;
    }
    jmethodID methodId = env->GetStaticMethodID(uuid_class, "randomUUID", "()Ljava/util/UUID;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("randomUUID() Failed.");
        env->DeleteLocalRef(uuid_class);
        return NULL;
    }
    jobject uuid_object = env->CallStaticObjectMethod(uuid_class, methodId);
    if(env->ExceptionOccurred() || uuid_object == NULL)
    {
        LOGE("randomUUID() Failed.");
        env->DeleteLocalRef(uuid_class);
        return NULL;
    }
    env->DeleteLocalRef(uuid_class);

    uuid_class = env->GetObjectClass(uuid_object);
    methodId = env->GetMethodID(uuid_class, "toString", "()Ljava/lang/String;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("toString() Failed.");
        env->DeleteLocalRef(uuid_class);
        env->DeleteLocalRef(uuid_object);
        return NULL;
    }
    env->DeleteLocalRef(uuid_class);
    jstring jstr_uuid = (jstring)env->CallObjectMethod(uuid_object, methodId);
    if(env->ExceptionOccurred() || jstr_uuid == NULL)
    {
        LOGE("toString() Failed.");
        env->DeleteLocalRef(uuid_object);
        return NULL;
    }
    env->DeleteLocalRef(uuid_object);

    const char *suuid = env->GetStringUTFChars(jstr_uuid, 0);
    char *str_uuid = (char*)malloc(strlen(suuid)+1);
    assert(str_uuid != NULL);
    memcpy(str_uuid, suuid, strlen(suuid));
    str_uuid[strlen(suuid)] = '\0';
    env->ReleaseStringUTFChars(jstr_uuid, suuid);

    //LOGE("uuid generated: %s", str_uuid);
    return str_uuid;
}

// 将单个字符转成数字
int HexChar2Int(const char &c)
{
    assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    if(c >= '0' && c <= '9') return (c-'0');
    if(c >= 'a' && c <= 'f') return (c-'a'+10);
    return -1;
}

// 将16进制字符串转成字节数组
// 返回结果需要手动释放
unsigned char* HexString2UcharArray(const char *str_hex)
{
    assert(str_hex != NULL);

    int len = strlen(str_hex);
    assert((len % 2 == 0) && (len != 0));

    unsigned char *result = (unsigned char*)malloc(len/2);
    assert(result != NULL);

    for(int i = 0, j = 0; i < len; i += 2, j++)
    {
        int high = HexChar2Int(str_hex[i]);
        int low = HexChar2Int(str_hex[i+1]);
        unsigned char ret = ((high & 0x0f) << 4) | (low & 0x0f);
        result[j] = ret;
    }

    return result;
}

// 计算字符串的md5值，英文字母用小写
// 返回结果需要手动释放
char* generate_md5(const char* str_input, int len)
{
    assert(str_input != NULL);

    unsigned char *md5_ret = (unsigned char*)malloc(MD5_DIGEST_LENGTH);
    assert(md5_ret != NULL);

    MD5((const unsigned char*)str_input, len, md5_ret);

    char *result = (char*)malloc(MD5_DIGEST_LENGTH*2+1);
    assert(result != NULL);

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        sprintf(result+2*i, "%02x", md5_ret[i]);
    }
    result[MD5_DIGEST_LENGTH*2] = '\0';
    free(md5_ret);

    return result;
}

// 对字节数组进行base64编码
// 返回结果需要手动释放
char* encode_base64(const unsigned char *str_input, int len)
{
    BIO *bmem = BIO_new(BIO_s_mem());
    assert(bmem != NULL);

    BIO *b64 = BIO_new(BIO_f_base64());
    assert(b64 != NULL);

    b64 = BIO_push(b64, bmem);
    BIO_write(b64, str_input, len);
    BIO_flush(b64);

    BUF_MEM *bptr;
    BIO_get_mem_ptr(b64, &bptr);
    assert(bptr != NULL);

    char *result = (char*)malloc(bptr->length+1);
    assert(result != NULL);
    memcpy(result, bptr->data, bptr->length);
    result[bptr->length] = '\0';

    BIO_free_all(b64);

    return result;
}

// 加密字符串，加密算法为"AES/CBC/PKCS5Padding"，密钥长度为128
// 返回结果需要手动释放
char* encrypt_string(const char *str2encrypt, const char *str_key)
{
    assert(str2encrypt != NULL && str_key != NULL);

    int nLen = strlen(str2encrypt);
    int nBlock = nLen / AES_BLOCK_SIZE + 1;
    int nTotal = nBlock * AES_BLOCK_SIZE;

    // KCS5Padding：填充的原则是，如果长度少于16个字节，需要补满16个字节，补(16-len)个(16-len)
    // 例如：huguPozhen这个节符串是9个字节，16-9=7，补满后如：huguozhen+7个十进制的7
    // 如果字符串长度正好是16字节，则需要再补16个字节的十进制的16
    char *str_padded = (char*)malloc(nTotal);
    assert(str_padded != NULL);
    int nNumber;
    if(nLen % 16 > 0)
        nNumber = nTotal - nLen;
    else
        nNumber = 16;
    memset(str_padded, nNumber, nTotal);
    memcpy(str_padded, str2encrypt, nLen);

    AES_KEY aes;
    const int nKeyLenght = 128;
    unsigned char iv[AES_BLOCK_SIZE];
    memset(iv, 0x00, AES_BLOCK_SIZE);
    unsigned char *key = HexString2UcharArray(str_key);

    assert(AES_set_encrypt_key(key, nKeyLenght, &aes) >= 0);
    free(key);

    unsigned char *encrypted = (unsigned char*)malloc(nLen+1);
    assert(encrypted != NULL);

    AES_cbc_encrypt((unsigned char*)str_padded, encrypted, nBlock*16, &aes, (unsigned char*)iv, AES_ENCRYPT);
    free(str_padded);

    char *result = encode_base64(encrypted, nLen+1);
    free(encrypted);
    return result;
}

// 加密数据
// 返回结果需要手动释放
char* obtain_encrypted_string(const char *str_json, const char *md5_rule)
{
    assert(str_json != NULL && md5_rule != NULL);

    char *md5_1 = generate_md5(md5_rule, strlen(md5_rule));
    char *str_key = generate_md5(md5_1, strlen(md5_1));
    free(md5_1);

    char *result = encrypt_string(str_json, str_key);
    free(str_key);
    return result;
}

bool check_package_keystore_apikey(const std::string &input_pkg, const std::string &input_sha1, const std::string &input_apikey)
{
    if(input_pkg == global_str_turing_package_name && input_sha1 == global_str_turing_keystore_sha1 && input_apikey == global_str_turing_apikey)
        return true;
    return false;
}

// 通过java反射获取app的包名和keystore的SHA1
void generate_package_sha1( JNIEnv* env,
                            jobject &context_object,
                            std::string &package_name,
                            std::string &keystore_sha1 )
{
    package_name = "";
    keystore_sha1 = "";
    const char HexCode[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
    /** java code
        //获取包管理器
        PackageManager packageManager = context.getPackageManager();
    */
    jclass context_class = env->GetObjectClass(context_object);
    jmethodID methodId = env->GetMethodID(context_class, "getPackageManager", "()Landroid/content/pm/PackageManager;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("getPackageManager() Failed.");
        env->DeleteLocalRef(context_class);
        return;
    }
    jobject package_manager_object = env->CallObjectMethod(context_object, methodId);
    if(env->ExceptionOccurred() || package_manager_object == NULL)
    {
        LOGE("getPackageManager() Failed.");
        env->DeleteLocalRef(context_class);
        return;
    }
    //left: context_class package_manager_object

    /** java code
        //获取包名
        String packageName = context.getPackageName();
    */
    methodId = env->GetMethodID(context_class, "getPackageName", "()Ljava/lang/String;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("getPackageName() Failed.");
        env->DeleteLocalRef(context_class);
        env->DeleteLocalRef(package_manager_object);
        return;
    }
    jstring package_name_string = (jstring)env->CallObjectMethod(context_object, methodId);
    if(env->ExceptionOccurred() || package_name_string == NULL)
    {
        LOGE("getPackageName() Failed.");
        env->DeleteLocalRef(context_class);
        env->DeleteLocalRef(package_manager_object);
        return;
    }
    env->DeleteLocalRef(context_class);
    //left: package_manager_object package_name_string

    const char *str_pkg = env->GetStringUTFChars(package_name_string, 0);
    //LOGE("package: %s", str_pkg);
    std::string str_pkg_name = std::string(str_pkg);
    package_name.swap(str_pkg_name);
    env->ReleaseStringUTFChars(package_name_string, str_pkg);
    //left: package_manager_object

    /** java code
        //获得包信息
        //public static final int GET_SIGNATURES = 0x00000040;
        PackageInfo pis = packageManager.getPackageInfo(packageName, PackageManager.GET_SIGNATURES);
    */
    jclass pack_manager_class = env->GetObjectClass(package_manager_object);
    methodId = env->GetMethodID(pack_manager_class, "getPackageInfo", "(Ljava/lang/String;I)Landroid/content/pm/PackageInfo;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("getPackageInfo() Failed.");
        env->DeleteLocalRef(pack_manager_class);
        env->DeleteLocalRef(package_manager_object);
        return;
    }
    env->DeleteLocalRef(pack_manager_class);
    jobject package_info_object = env->CallObjectMethod(package_manager_object, methodId, package_name_string, 0x40);
    if(env->ExceptionOccurred() || package_info_object == NULL)
    {
        LOGE("getPackageInfo() Failed.");
        env->DeleteLocalRef(package_manager_object);
        return;
    }
    env->DeleteLocalRef(package_manager_object);
    //left: package_info_object

    /** java code
        //获得签名
        Signature[] signs = pis.signatures;
        //获得签名数组的第一位
        Signature sign = signs[0];
    */
    jclass package_info_class = env->GetObjectClass(package_info_object);
    jfieldID fieldId = env->GetFieldID(package_info_class, "signatures", "[Landroid/content/pm/Signature;");
    if(env->ExceptionOccurred() || fieldId == NULL)
    {
        LOGE("get signatures Failed.");
        env->DeleteLocalRef(package_info_class);
        env->DeleteLocalRef(package_info_object);
        return;
    }
    env->DeleteLocalRef(package_info_class);
    jobjectArray signature_object_array = (jobjectArray)env->GetObjectField(package_info_object, fieldId);
    if(env->ExceptionOccurred() || signature_object_array == NULL)
    {
        LOGE("get signatures Failed.");
        env->DeleteLocalRef(package_info_object);
        return;
    }
    env->DeleteLocalRef(package_info_object);
    //left: signature_object_array

    /** java code
        //获得签名数组的第一位
        Signature sign = signs[0];
    */
    jobject signature_object = env->GetObjectArrayElement(signature_object_array, 0);
    env->DeleteLocalRef(signature_object_array);
    //left: signature_object

    /** java code
        //转成字节数组
        byte[] signBytes = sign.toByteArray();
    */
    jclass signature_class = env->GetObjectClass(signature_object);
    methodId = env->GetMethodID(signature_class, "toByteArray", "()[B");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("toByteArray() Failed.");
        env->DeleteLocalRef(signature_class);
        env->DeleteLocalRef(signature_object);
        return;
    }
    env->DeleteLocalRef(signature_class);
    jbyteArray signature_byte = (jbyteArray)env->CallObjectMethod(signature_object, methodId);
    if(env->ExceptionOccurred() || signature_byte == NULL)
    {
        LOGE("toByteArray() Failed.");
        env->DeleteLocalRef(signature_object);
        return;
    }
    env->DeleteLocalRef(signature_object);
    //left: signature_byte

    /** java code
        ByteArrayInputStream byteIn = new ByteArrayInputStream(signBytes);
    */
    jclass byte_array_input_class = env->FindClass("java/io/ByteArrayInputStream");
    if(env->ExceptionOccurred() || byte_array_input_class == NULL)
    {
        LOGE("find ByteArrayInputStream Failed.");
        env->DeleteLocalRef(signature_byte);
        return;
    }
    methodId = env->GetMethodID(byte_array_input_class, "<init>", "([B)V");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("find ByteArrayInputStream Failed.");
        env->DeleteLocalRef(signature_byte);
        env->DeleteLocalRef(byte_array_input_class);
        return;
    }
    jobject byte_array_input = env->NewObject(byte_array_input_class, methodId, signature_byte);
    env->DeleteLocalRef(signature_byte);
    env->DeleteLocalRef(byte_array_input_class);
    //left: byte_array_input

    /** java code
        //获得X.509证书工厂
        CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
    */
    jclass certificate_factory_class = env->FindClass("java/security/cert/CertificateFactory");
    if(env->ExceptionOccurred() || certificate_factory_class == NULL)
    {
        LOGE("find CertificateFactory Failed.");
        env->DeleteLocalRef(byte_array_input);
        return;
    }
    methodId = env->GetStaticMethodID(certificate_factory_class, "getInstance", "(Ljava/lang/String;)Ljava/security/cert/CertificateFactory;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("getInstance() Failed.");
        env->DeleteLocalRef(byte_array_input);
        env->DeleteLocalRef(certificate_factory_class);
        return;
    }
    jstring x_509_jstring = env->NewStringUTF("X.509");
    jobject cert_factory = env->CallStaticObjectMethod(certificate_factory_class, methodId, x_509_jstring);
    if(env->ExceptionOccurred() || cert_factory == NULL)
    {
        LOGE("getInstance() Failed.");
        env->DeleteLocalRef(byte_array_input);
        env->DeleteLocalRef(certificate_factory_class);
        env->DeleteLocalRef(x_509_jstring);
        return;
    }
    env->DeleteLocalRef(x_509_jstring);
    //left: byte_array_input certificate_factory_class cert_factory

    /** java code
        //获取X509证书
        X509Certificate cert = (X509Certificate) certFactory.generateCertificate(byteIn);
    */
    methodId = env->GetMethodID(certificate_factory_class, "generateCertificate", ("(Ljava/io/InputStream;)Ljava/security/cert/Certificate;"));
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("generateCertificate() Failed.");
        env->DeleteLocalRef(byte_array_input);
        env->DeleteLocalRef(cert_factory);
        env->DeleteLocalRef(certificate_factory_class);
        return;
    }
    jobject x509_cert = env->CallObjectMethod(cert_factory, methodId, byte_array_input);
    env->DeleteLocalRef(byte_array_input);
    env->DeleteLocalRef(cert_factory);
    if(env->ExceptionOccurred() || x509_cert == NULL)
    {
        LOGE("generateCertificate() Failed.");
        env->DeleteLocalRef(certificate_factory_class);
        return;
    }
    env->DeleteLocalRef(certificate_factory_class);
    //left: x509_cert

    /** java code
        //获取证书发行者SHA1
        MessageDigest sha1 = MessageDigest.getInstance("SHA1");
    */
    jclass message_digest_class = env->FindClass("java/security/MessageDigest");
    if(env->ExceptionOccurred() || message_digest_class == NULL)
    {
        LOGE("find MessageDigest Failed.");
        env->DeleteLocalRef(x509_cert);
        return;
    }
    methodId = env->GetStaticMethodID(message_digest_class, "getInstance", "(Ljava/lang/String;)Ljava/security/MessageDigest;");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("getInstance() Failed.");
        env->DeleteLocalRef(x509_cert);
        env->DeleteLocalRef(message_digest_class);
        return;
    }
    jstring sha1_jstring = env->NewStringUTF("SHA1");
    jobject sha1_digest = env->CallStaticObjectMethod(message_digest_class, methodId, sha1_jstring);
    if(env->ExceptionOccurred() || sha1_digest == NULL)
    {
        LOGE("getInstance() Failed.");
        env->DeleteLocalRef(x509_cert);
        env->DeleteLocalRef(message_digest_class);
        env->DeleteLocalRef(sha1_jstring);
        return;
    }
    env->DeleteLocalRef(sha1_jstring);
    //left: x509_cert message_digest_class sha1_digest

    /** java code
        byte[] certByte = cert.getEncoded();
    */
    jclass x509_cert_class = env->GetObjectClass(x509_cert);
    methodId = env->GetMethodID(x509_cert_class, "getEncoded", "()[B");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("getEncoded() Failed.");
        env->DeleteLocalRef(x509_cert);
        env->DeleteLocalRef(message_digest_class);
        env->DeleteLocalRef(sha1_digest);
        env->DeleteLocalRef(x509_cert_class);
        return;
    }
    env->DeleteLocalRef(x509_cert_class);
    jbyteArray cert_byte = (jbyteArray)env->CallObjectMethod(x509_cert, methodId);
    if(env->ExceptionOccurred() || cert_byte == NULL)
    {
        LOGE("getEncoded() Failed.");
        env->DeleteLocalRef(x509_cert);
        env->DeleteLocalRef(message_digest_class);
        env->DeleteLocalRef(sha1_digest);
        return;
    }
    env->DeleteLocalRef(x509_cert);
    //left: message_digest_class sha1_digest cert_byte

    /** java code
        byte[] bs = sha1.digest(certByte);
    */
    methodId = env->GetMethodID(message_digest_class, "digest", "([B)[B");
    if(env->ExceptionOccurred() || methodId == NULL)
    {
        LOGE("digest() Failed.");
        env->DeleteLocalRef(message_digest_class);
        env->DeleteLocalRef(sha1_digest);
        env->DeleteLocalRef(cert_byte);
        return;
    }
    jbyteArray sha1_byte = (jbyteArray)env->CallObjectMethod(sha1_digest, methodId, cert_byte);
    if(env->ExceptionOccurred() || sha1_byte == NULL)
    {
        LOGE("digest() Failed.");
        env->DeleteLocalRef(message_digest_class);
        env->DeleteLocalRef(sha1_digest);
        env->DeleteLocalRef(cert_byte);
        return;
    }
    env->DeleteLocalRef(message_digest_class);
    env->DeleteLocalRef(sha1_digest);
    env->DeleteLocalRef(cert_byte);
    //left: sha1_byte

    //toHexString
    jsize array_size = env->GetArrayLength(sha1_byte);
    jbyte* sha1 = env->GetByteArrayElements(sha1_byte, NULL);
    char* hex_sha1 = new char[array_size*3];
    for(int i = 0; i <array_size; i++)
    {
        hex_sha1[3*i+0] = HexCode[((unsigned char)sha1[i])/16];
        hex_sha1[3*i+1] = HexCode[((unsigned char)sha1[i])%16];
        hex_sha1[3*i+2] = ':';
    }
    hex_sha1[array_size*3-1] = '\0';
    env->ReleaseByteArrayElements(sha1_byte, sha1, JNI_ABORT);
    std::string str_keystore_sha1 = std::string(hex_sha1);
    keystore_sha1.swap(str_keystore_sha1);
    delete []hex_sha1;

    return;
}

// function 2 receive data 
size_t recvData(void *buffer, size_t sz, size_t nmemb, void *userp)
{
    int wr_index = 0;
    int segsize = sz * nmemb;
    if(wr_index + segsize > RECV_BUFFER_SIZE)
    {
        *(int *)userp = 1;
        return 0;
    }

    memcpy((void*)&recv_buff[wr_index], buffer, (size_t)segsize);
    wr_index += segsize;
    recv_buff[wr_index] = 0;

    return segsize;
}

// main function 2 post http or https request with json format data
int postJsonHttpRequest(const std::string &strUrl, const std::string &strJson, const int &requestType, 
						const std::string &caFile, const bool &useCA=true, int timeout=5)
{
    try
    {
        CURL *pCurl = NULL;
        curl_global_init(CURL_GLOBAL_ALL);
        pCurl = curl_easy_init();

        if(NULL != pCurl)
        {
            // 设置http发送的内容类型为JSON
            curl_slist *plist = curl_slist_append(NULL, "Content-Type:application/json;charset=UTF-8");
            // 设置超时时间
            curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, timeout);
            curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, plist);
            curl_easy_setopt(pCurl, CURLOPT_URL, strUrl.c_str());
            // 设置要POST的JSON数据
            curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, strJson.c_str());
            // 设置接受数据的回调函数
            curl_easy_setopt(pCurl, CURLOPT_WRITEFUNCTION, recvData);
            // 如果是https请求则要添加证书
            if(requestType == POST_TYPE_HTTPS)
            {
            	if(useCA)
            	{
	                curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYHOST, 2);
	                curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYPEER, true);
            	}
            	else 
            	{
            		curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYHOST, 0);
	                curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYPEER, false);
            	}
                curl_easy_setopt(pCurl, CURLOPT_CAINFO, caFile.c_str());
            }
            // 发送请求
            CURLcode res = curl_easy_perform(pCurl);
            if(res != CURLE_OK)
            {
                LOGE("network failed: %d", res /*curl_easy_strerror(res)*/);
                curl_easy_cleanup(pCurl);
                curl_global_cleanup();
                return -3;
            }
            // always cleanup
            curl_easy_cleanup(pCurl);
        }
        else
        {
        	curl_global_cleanup();
            return -2;
        }
    }
    catch(std::exception &ex)
    {
    	curl_global_cleanup();
        return -1;
    }

    curl_global_cleanup();
    return 1;
}

// self parse string, format: {"ret":0,"time":"2017-04-21 17:36:31"}
// get [ret] value
int getRetResult(const char *buff, int len)
{
    int result = 404;  // 'ret' not found 
    if(buff == NULL || len <= 0)
        return result;

    std::string pattern = "\"ret\"";
    std::string str_buff = std::string(buff);
    int len_patt = pattern.length();

    for(int i = 0; i < len-len_patt; i++)
    {
        if(pattern == str_buff.substr(i, len_patt))
        {
            std::string rest = str_buff.substr(i+len_patt);
            int colon_pos = rest.find_first_of(':');
            int comma_pos = rest.find_first_of(',');
            if(comma_pos==-1)
                comma_pos = rest.find_first_of('}');
            result = atoi(rest.substr(colon_pos+1, comma_pos-colon_pos-1).c_str());
            break;
        }
    }

    return result;
}

// self parse string, format: {"ret":0,"userid":"sfasdccxvxcvcx"}
// get [userid] value
std::string getUseridResult(const char *buff, int len)
{
    std::string result = "";
    if(buff == NULL || len <= 0)
        return result;

    std::string pattern = "\"userid\"";
    std::string str_buff = std::string(buff);
    int len_patt = pattern.length();

    for(int i = 0; i < len-len_patt; i++)
    {
        if(pattern == str_buff.substr(i, len_patt))
        {
            std::string rest = str_buff.substr(i+len_patt);
            int colon_pos = rest.find_first_of(':');
            int comma_pos = rest.find_first_of(',');
            if(comma_pos==-1)
                comma_pos = rest.find_first_of('}');
            std::string tmp = rest.substr(colon_pos+1, comma_pos-colon_pos-1);
            int first_quote = tmp.find_first_of('\"');
            int last_quote = tmp.find_last_of('\"');
            result = tmp.substr(first_quote, last_quote-first_quote+1);
            break;
        }
    }

    return result;
}

// get userid from server 
int obtainUserid_https(const char *data)
{
    memset(recv_buff, 0, sizeof(char)*RECV_BUFFER_SIZE);

    std::string strJson = std::string(data);
    int result = postJsonHttpRequest(URL_OBTAIN_USERID, strJson, POST_TYPE_HTTPS, CA_GET_USERID);

    return result;
}

// verify (apikey, userid) pair 
int verifyApikey_https(const char *apikey, const char *userid)
{
    memset(recv_buff, 0, sizeof(char)*RECV_BUFFER_SIZE);

	std::string strJson = "{\"key\":\""+std::string(apikey)+"\",\"userid\":"+std::string(userid)+",\"sdk_type\":\"VISUAL_VERSION\",\"service_identify\":\"validityVerification\"}";
    int result = postJsonHttpRequest(URL_VERIFY_APIKEY, strJson, POST_TYPE_HTTPS, CA_VERIFY_KEY, false);

	return result;
}

/*
-1:network error
-2:get userid failed
 0:invalid apikey
 1:succeed
*/
int checkValidation(const char *apikey, const char *data)
{
    // get userid
	int res = obtainUserid_https(data);
    if(res < 0)
    {
    	set_status(-1);
        return -1;  // network error
    }
    else
    {
        //LOGE(recv_buff);
        if(getRetResult(recv_buff, strlen(recv_buff)) != 0)
        {
        	set_status(-2);
            return -2;  // get userid failed
        }
    }

    // extract userid
	std::string userid = getUseridResult(recv_buff, strlen(recv_buff));  // with quote, like "84277487"

	// verify apikey
	res = verifyApikey_https(apikey, userid.c_str());
    if(res < 0)
    {
    	set_status(-1);
        return -1;  // network error
    }
    else
    {
        //LOGE(recv_buff);
        if(getRetResult(recv_buff, strlen(recv_buff)) != 0)
        {
        	set_status(0);
            return 0;  // invalid apikey
        }
    }

	set_status(1);
    return 1;
}

/*
-1:network error
-2:get userid failed
-3:invalid signature
-4:miss userid file
 0:invalid apikey
 1:succeed
*/
JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_jniInit
  (JNIEnv* env, jclass, jobject context, jstring japikey, jstring jkeysec)
{
    // set userid file path
    std::string VALIDATION_USER_PATH = "/sdcard/.";
    set_status(-2);

    // get api key
    const char *apikey = env->GetStringUTFChars(japikey, 0);
    std::string api_key = std::string(apikey);
    env->ReleaseStringUTFChars(japikey, apikey);

    // get api key secret
    const char *keysec = env->GetStringUTFChars(jkeysec, 0);
    std::string key_sec = std::string(keysec);
    env->ReleaseStringUTFChars(jkeysec, keysec);

    // get uuid
    char *uuid = generate_uuid(env);
    std::string str_uuid = std::string(uuid);
    free(uuid);

    // get timestamp
    uuid = generate_uuid(env);
    std::string date_str = std::string(uuid);
    free(uuid);

    // construct original json object
    std::string md5_rule = key_sec + date_str + api_key;
    std::string json_obj = "{\"uniqueId\":\""+str_uuid+"\",\"key\":\""+api_key+"\",\"os_version\":\"1.5\"}";
    //LOGE("json_obj: %s", json_obj.c_str());

    // get encrypted string
    char *data = obtain_encrypted_string(json_obj.c_str(), md5_rule.c_str());
    std::string enc_data = std::string(data);
    free(data);

    // remove unexpected '\n'
    for(string::iterator it = enc_data.begin(); it != enc_data.end(); )
    {
        if(*it == '\n')
            it = enc_data.erase(it);
        else
            it++;
    }
    
    // construct request data
    std::string req_json = "{\"timestamp\":\""+date_str+"\",\"key\":\""+api_key+"\",\"data\":\""+enc_data+"\"}";
    //LOGE("req_json: %s", req_json.c_str());
    
    // get package name and keystore sha1
    std::string package_name;
    std::string keystore_sha1;
    generate_package_sha1(env, context, package_name, keystore_sha1);

    // check validation of package name, keystore sha1 and api key
    bool validation_result = check_package_keystore_apikey(package_name, keystore_sha1, api_key);

    /*
    LOGE("defined api key: %s", global_str_turing_apikey.c_str());
    LOGE("current api key: %s", api_key.c_str());
    LOGE("defined package: %s", global_str_turing_package_name.c_str());
    LOGE("current package: %s", package_name.c_str());
    LOGE("defined sha1: %s", global_str_turing_keystore_sha1.c_str());
    LOGE("current sha1: %s", keystore_sha1.c_str());
    LOGE("check result: %s", validation_result?"true":"false");
    */

    if(!validation_result)
    {
        // invalid package or keystore or apikey
        set_status(-3);
        return -3;
    }

    // construct userid filename
    VALIDATION_USER_PATH += package_name;

    // check userid file
    std::fstream inf_user;
    inf_user.open(VALIDATION_USER_PATH, std::ios::in);
    if(inf_user)
    {
        //LOGE("userid from file");
        inf_user.close();
        set_status(1);
        return 1;
    }

    // request userid
    const char *data4userid = req_json.c_str();
    int ret = obtainUserid_https(data4userid);
    if(ret < 0)
    {
        set_status(-1);
        return -1;  // network error
    }
    else
    {
        //LOGE(recv_buff);
        if(getRetResult(recv_buff, strlen(recv_buff)) != 0)
        {
            set_status(-2);
            return -2;  // get userid failed
        }
    }

    // save userid
    std::string userid = getUseridResult(recv_buff, strlen(recv_buff));  // with quote, like "84277487"
    std::ofstream otf_user(VALIDATION_USER_PATH);
    otf_user << "0xf0b1" << userid;
    otf_user.close();

    //LOGE("userid generated");
    set_status(1);
    return 1;
}


/*
-1:network error
-2:get userid failed
-3:invalid signature
-4:miss userid file
 0:invalid apikey
 1:succeed
*/
JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_jniInit_backup
  (JNIEnv* env, jclass, jobject context, jstring japikey, jstring jdata4userid)
{
    std::string VALIDATION_USER_PATH = "/sdcard/.";
    // get api key
    const char *apikey = env->GetStringUTFChars(japikey, 0);
    std::string api_key = std::string(apikey);
    env->ReleaseStringUTFChars(japikey, apikey);

    // get package name and keystore sha1
    std::string package_name;
    std::string keystore_sha1;
    generate_package_sha1(env, context, package_name, keystore_sha1);

    // check validation of package name, keystore sha1 and api key
    bool validation_result = check_package_keystore_apikey(package_name, keystore_sha1, api_key);

    if(!validation_result)
    {
        // invalid package or keystore or apikey
        set_status(-3);
        return -3;
    }

    // construct userid filename
    VALIDATION_USER_PATH += package_name;

    // check userid file
    std::fstream inf_user;
    inf_user.open(VALIDATION_USER_PATH, std::ios::in);
    if(inf_user)
    {
        inf_user.close();
        set_status(1);
        return 1;
    }

    // request userid
    const char *data4userid = env->GetStringUTFChars(jdata4userid, 0);
    int ret = obtainUserid_https(data4userid);
    if(ret < 0)
    {
        env->ReleaseStringUTFChars(jdata4userid, data4userid);
        set_status(-1);
        return -1;  // network error
    }
    else
    {
        //LOGE(recv_buff);
        if(getRetResult(recv_buff, strlen(recv_buff)) != 0)
        {
            env->ReleaseStringUTFChars(jdata4userid, data4userid);
            set_status(-2);
            return -2;  // get userid failed
        }
    }
    env->ReleaseStringUTFChars(jdata4userid, data4userid);

    // save userid
    std::string userid = getUseridResult(recv_buff, strlen(recv_buff));  // with quote, like "84277487"
    std::ofstream otf_user(VALIDATION_USER_PATH);
    otf_user << "0xf0b1" << userid;
    otf_user.close();

    set_status(1);
    return 1;
}


/*
JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_checkValidation
  (JNIEnv *env, jclass, jstring japikey, jstring jdata4userid)
{
    const char *apikey = env->GetStringUTFChars(japikey, 0);
    const char *data4userid = env->GetStringUTFChars(jdata4userid, 0);
    //LOGE(apikey);
    int ret = checkValidation(apikey, data4userid);
    env->ReleaseStringUTFChars(japikey, apikey);
   	env->ReleaseStringUTFChars(jdata4userid, data4userid);
   	return ret;
}
*/


char* symbol;
char* params;
jsize params_len;

JNIEXPORT jint JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_decodeNN
  (JNIEnv *env, jclass, jbyteArray jsymbol, jbyteArray jparams)
{
	int ret = get_status();
	if(ret != 1) 
	{
		LOGE("failure status: %d", ret);
        return ret;
	}

    int len = env->GetArrayLength (jparams);
    char* buf = new char[len];
    params_len=len;
    env->GetByteArrayRegion (jparams, 0, len, reinterpret_cast<jbyte*>(buf));
    params = new char[len];

    string keyStr = "tulingdemimahaha"; //密码
    byte key[16];
    charToByte(key, keyStr.c_str());
    word w[4*(Nr+1)];
    KeyExpansion(key, w);
    byte plain[16];

    ofstream out;
    out.open("/sdcard/encrypted", ios::binary); //将密文存到sdcard
    out.write((char*)buf, len);
    out.close();
    ifstream in;
    in.open("/sdcard/encrypted", ios::binary);
    long int startidx=0;
    bitset<128> data;
    int position=0;
    while(in.read((char*)&data, sizeof(data)))
    {
        divideToByte(plain, data);
        if(++position<1000)
            decrypt(plain, w);
        for(int d=0;d<16;d++)
        {
            params[d+startidx] = static_cast<char>(  plain[15-d].to_ulong() );
        }
        startidx=startidx+16;
    }

    // delete[] buf;
    env->ReleaseByteArrayElements(jparams, reinterpret_cast<jbyte*>(buf), 0);
    in.close();

    // jbyte* symbol = env->GetByteArrayElements(jsymbol, 0);
    len = env->GetArrayLength (jsymbol);
    buf = new char[len];

    env->GetByteArrayRegion (jsymbol, 0, len, reinterpret_cast<jbyte*>(buf));
    symbol = new char[len];
    out.open("/sdcard/encrypted-json", ios::binary); //将密文存到sdcard
    out.write((char*)buf, len);
    out.close();

    in.open("/sdcard/encrypted-json", ios::binary);
    startidx=0;
    // bitset<128> data;

    while(in.read((char*)&data, sizeof(data)))
    {
        divideToByte(plain, data);

        decrypt(plain, w);
        for(int d=0;d<16;d++)
        {
            symbol[d+startidx] = static_cast<char>(  plain[15-d].to_ulong() );
        }
        startidx=startidx+16;
    }

    // delete[] buf;
    env->ReleaseByteArrayElements(jsymbol, reinterpret_cast<jbyte*>(buf), 0);
    in.close();
    return 1;
}

JNIEXPORT jlong JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_createPredictor
  (JNIEnv *env, jclass, jint devType, jint devId, jobjectArray jkeys, jobjectArray jshapes)
{
    PredictorHandle handle = 0;

    std::vector<std::pair<jstring, const char *>> track;
    std::vector<const char *> keys;
    for (int i=0; i<env->GetArrayLength(jkeys); i++) {
    	  jstring js = (jstring) env->GetObjectArrayElement(jkeys, i);
    	  const char *s = env->GetStringUTFChars(js, 0);
    keys.emplace_back(s);
    track.emplace_back(js, s);
    }

    std::vector<mx_uint> index{0};
    std::vector<mx_uint> shapes;
    for (int i=0; i<env->GetArrayLength(jshapes); i++) {
    	  jintArray jshape = (jintArray) env->GetObjectArrayElement(jshapes, i);
    jsize shape_len = env->GetArrayLength(jshape);
    jint *shape = env->GetIntArrayElements(jshape, 0);

    index.emplace_back(shape_len);
    for (int j=0; j<shape_len; ++j) shapes.emplace_back((mx_uint)shape[j]);
    env->ReleaseIntArrayElements(jshape, shape, 0);
    }


    if (MXPredCreate((const char *)symbol, (const char *)params, params_len, devType, devId, (mx_uint)keys.size(), &(keys[0]), &(index[0]), &(shapes[0]), &handle) < 0) {
    	jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
    	env->ThrowNew(MxnetException, MXGetLastError());
    }	 

    for (auto& t: track) {
    	env->ReleaseStringUTFChars(t.first, t.second);
    }
    delete[] params;
    delete[] symbol;
    return (jlong)handle;
}

JNIEXPORT void JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_nativeFree
  (JNIEnv *, jclass, jlong h)
{
	PredictorHandle handle = (PredictorHandle)h;	
	MXPredFree(handle);
}

JNIEXPORT jfloatArray JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_nativeGetOutput
  (JNIEnv *env, jclass, jlong h, jint index)
{
	PredictorHandle handle = (PredictorHandle)h;	

	mx_uint *shape = 0;
	mx_uint shape_len;
	if (MXPredGetOutputShape(handle, index, &shape, &shape_len) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}

	size_t size = 1;
	for (mx_uint i=0; i<shape_len; ++i) size *= shape[i];

	std::vector<float> data(size);
	if (MXPredGetOutput(handle, index, &(data[0]), size) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}
	
	jfloatArray joutput = env->NewFloatArray(size);
    jfloat *out = env->GetFloatArrayElements(joutput, NULL);

    for (int i=0; i<size; i++) out[i] = data[i];
    env->ReleaseFloatArrayElements(joutput, out, 0);

	return joutput;
}

JNIEXPORT void JNICALL Java_org_dmlc_mxnet_PredictorEncrypt_nativeForward
  (JNIEnv *env, jclass, jlong h, jstring jkey, jfloatArray jinput)
{
	PredictorHandle handle = (PredictorHandle)h;	
	const char *key = env->GetStringUTFChars(jkey, 0);
	jfloat* input = env->GetFloatArrayElements(jinput, 0);
	jsize input_len = env->GetArrayLength(jinput);

	if (MXPredSetInput(handle, key, input, input_len) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}

	env->ReleaseStringUTFChars(jkey, key);
	env->ReleaseFloatArrayElements(jinput, input, 0);
	if (MXPredForward(handle) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}
}



