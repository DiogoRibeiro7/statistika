{
    "targets": [
        {
            "target_name": "fortran_special",
            "variables": {"use_lapack%": 0},
            "sources": ["native/cpp/addon.cpp"],
            "include_dirs": ["<!@(node -p \"require('node-addon-api').include\")"],
            "conditions": [
                [
                    "OS=='win'",
                    {
                        "libraries": [
                            "<(module_root_dir)/build/Release/libfortran_native.dll.a"
                        ]
                    }
                ],
                [
                    "use_lapack=='1' and OS!='win'",
                    {
                        "libraries": [
                            "<(module_root_dir)/native/fortran/special_functions.o",
                            "<(module_root_dir)/native/fortran/distributions.o",
                            "<(module_root_dir)/native/fortran/linalg.o",
                            "<(module_root_dir)/native/fortran/statistics.o",
                            "<(module_root_dir)/native/fortran/time_series.o",
                            "<(module_root_dir)/native/fortran/kalman.o",
                            "<(module_root_dir)/native/fortran/optimization.o",
                            "<(module_root_dir)/native/fortran/sampling.o",
                            "-llapack",
                            "-lblas",
                            "-lgfortran"
                        ]
                    }
                ],
                [
                    "use_lapack!='1' and OS!='win'",
                    {
                        "libraries": [
                            "<(module_root_dir)/native/fortran/special_functions.o",
                            "<(module_root_dir)/native/fortran/distributions.o",
                            "<(module_root_dir)/native/fortran/linalg.o",
                            "<(module_root_dir)/native/fortran/statistics.o",
                            "<(module_root_dir)/native/fortran/time_series.o",
                            "<(module_root_dir)/native/fortran/kalman.o",
                            "<(module_root_dir)/native/fortran/optimization.o",
                            "<(module_root_dir)/native/fortran/sampling.o",
                            "-lgfortran"
                        ]
                    }
                ]
            ],
            "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
            "cflags!": ["-fno-exceptions"],
            "cflags_cc!": ["-fno-exceptions"],
            "cflags_cc": ["-std=c++17"]
        }
    ]
}
