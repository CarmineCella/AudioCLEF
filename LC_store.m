function LC_storage (all_p, results)
    filename = sprintf ('%s/results/LifeClef2015_results_%s.txt', pwd, datestr (fix (clock)));
    fid = fopen (filename, 'w+');
    fprintf (fid, '%s\n\n', all_p);
    fprintf (fid, '%s\n\n', results);
    fclose (fid);
end
