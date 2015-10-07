function AC_store (all_p, results)
    filename = sprintf ('%s/results/AudioCLEF_results_%s.txt', pwd, datestr (fix (clock)));
    fid = fopen (filename, 'w+');
    fprintf (fid, '%s\n\n', all_p);
    fprintf (fid, '%s\n\n', results);
    fclose (fid);
end
