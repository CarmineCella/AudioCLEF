function LC_makefolders (foldr, dest)
    filenames = dir(strcat (foldr));
    filenames = filenames (3 : length(filenames));
        
    for iFile = 1 : length(filenames)    
        filename = strcat (foldr, '/',filenames(iFile).name);
        
        [~, nonly, extonly] = fileparts (filename);
        if (strcmp (extonly, '.xml') == false) 
           continue;
        end
        
        fprintf ('parsing %s\n', filename);
        n = parseXML (filename);
        ck = n.Children(6).Children.Data;
        fprintf ('class id = %s\n\n', ck);
        dfold = strcat (dest, '/', ck);
        mkdir (dfold);
        movefile (filename, dfold);
        auname = strcat (foldr, '/', nonly, '.wav');
        movefile (auname, dfold);
    end
end
