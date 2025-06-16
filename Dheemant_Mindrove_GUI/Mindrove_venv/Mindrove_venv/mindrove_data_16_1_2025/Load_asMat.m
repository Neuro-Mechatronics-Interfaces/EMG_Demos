csv_file = readtable("D:\CMU_NML\Mindrove_venv\mindrove_data_16_1_2025\Session_2_1.csv");
% disp(csv_file);
extracted_data = csv_file(:, 1:8);
Transposed_Chan_data = table2array(extracted_data)';
save('D:\CMU_NML\Mindrove_venv\mindrove_data_16_1_2025\Session_2_1_mat.mat', 'Transposed_Chan_data')