//
// move2clss.cpp
// 

// g++ move2clss.cpp -I /Librerie/tinyxml2-master/ -L /Librerie/tinyxml2-master/ -ltinyxml2 -o move2clss

#include "tinyxml2.h"
#include <stdexcept>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <limits.h>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>

using namespace std;
using namespace tinyxml2;

std::string removePath (const std::string& in) {
    size_t pos = std::string::npos;
    pos = in.find_last_of ("/");
    if (pos != std::string::npos) {
        return in.substr (pos + 1, in.size () - pos);
    } else return in;
}

std::string removeExtension (const std::string& in) {
    size_t pos = std::string::npos;
    pos = in.find_last_of (".");
    if (pos != std::string::npos) {
        return in.substr (0, pos);
    } else return in;
}

static void recurse_dir (const char* dir_name, const char* dest) {
    DIR * d;

    d = opendir (dir_name);

    if (! d) {
		stringstream msg;
		msg <<  "cannot open directory: " << dir_name << endl;
		throw runtime_error (msg.str ());
    }
    while (1) {
        struct dirent* entry;
        const char* d_name;

        entry = readdir (d);
        if (!entry) {
            break;
        }
        d_name = entry->d_name;

        if (! (entry->d_type & DT_DIR)) {
			string sname (d_name);
			if (sname.find (".xml")) {
				XMLDocument doc;
				doc.LoadFile (d_name);
				XMLElement* aud = doc.FirstChildElement ( "Audio" );
				if (aud) {
					XMLElement* clid = aud->FirstChildElement ("ClassId" );
					if (clid) {
						const char* title = clid->GetText ();
						cout << sname << " = " << title << endl;
						struct stat st = {0};

						stringstream outfoldr;
						outfoldr << dest << "/" << title;
						if (stat (outfoldr.str ().c_str (), &st) == -1) {
							mkdir (outfoldr.str ().c_str (), 0777);
						}

						string nopath = removePath (sname);
						string noext = removeExtension (sname);
						string noext_nopath = removeExtension (nopath);
						
						string xmldestname = dest + (string) title + "/" + nopath;
						string audorigname = noext + ".wav";
						string auddestname = dest + (string) title + "/" +  removePath (audorigname);

						//cout << sname << " " << xmldestname << " "<< audorigname << " " << auddestname << endl;

						ifstream xsource (sname.c_str (), ios::binary);
						ofstream xdest (xmldestname.c_str (), ios::binary);

						xdest << xsource.rdbuf();

						xsource.close();
						xdest.close();

						ifstream asource (audorigname.c_str (), ios::binary);
						ofstream adest (auddestname.c_str (), ios::binary);

						adest << asource.rdbuf();

						asource.close();
						adest.close();
					}
				} 
			}
		}

        if (entry->d_type & DT_DIR) {
            if (strcmp (d_name, "..") != 0 &&
                strcmp (d_name, ".") != 0) {
                int path_length;
                char path[PATH_MAX];
 
                path_length = snprintf (path, PATH_MAX,
                                        "%s/%s", dir_name, d_name);
				//                cout << path << endl;
                if (path_length >= PATH_MAX) {
					
                    throw runtime_error ("path length has got too long");
                }

                recurse_dir (path, dest);
            }
		}
    }
    if (closedir (d)) {
		stringstream msg;
		msg << "could not close " << dir_name << endl;
		throw runtime_error (msg.str ());
	}
}

int main (int argc, char* argv[]) {
	try {
		if (argc != 2) {
			throw runtime_error ("LC_mv2class dest_folder");
		}
		recurse_dir (".", argv[1]);
		
	} catch (exception& e) {
		cout << "error: " << e.what () << endl;
	} 
    
    return 0;
}

// EOF

