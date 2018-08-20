import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Get user dependent C/C++ header files',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--package_name', dest='package_name', help='java package name',
                        default="com.turing.vision.OCRDemo", type=str)
    parser.add_argument('--keystore_sha1', dest='keystore_sha1', help='user keystore sha1 value',
                        default="93:6F:9B:3B:74:3A:3F:B8:EC:DC:C3:34:8F:96:FB:BA:3F:2A:98:2C", type=str)
    parser.add_argument('--api_key', dest='api_key', help='api key applied from www.tuling123.com',
                        default="4782dd9f2577472a9eb0aa198cdcb421", type=str)
    argv = parser.parse_args()
    return argv


def write_header(in_package_name, in_api_key, in_keystore_sha1):
    f = open('predefined.h', 'w')
    f.write('#include <string>\n')
    f.write('\n')
    f.write('using namespace std;\n')
    f.write('\n\n')
    f.write('const std::string global_str_turing_package_name = \"%s\";\n' % in_package_name)
    f.write('\n')
    f.write('const std::string global_str_turing_apikey = \"%s\";\n' % in_api_key)
    f.write('\n')
    f.write('const std::string global_str_turing_keystore_sha1 = \"%s\";\n' % in_keystore_sha1)
    f.write('\n\n')
    f.close()


if __name__ == '__main__':
    args = parse_arguments()
    print args
    package_name = args.package_name.replace('\n', '')
    api_key = args.api_key.replace('\n', '')
    keystore_sha1 = args.keystore_sha1.replace('\n', '')
    write_header(package_name, api_key, keystore_sha1)
    print 'finish'



