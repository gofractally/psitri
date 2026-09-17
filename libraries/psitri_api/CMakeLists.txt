# Supported C++ API. Linking it never links the research engines or the legacy
# psitri implementation. Use engine-specific targets to narrow dependencies.
add_library(psitri_api_concepts INTERFACE)
add_library(psitri::concepts ALIAS psitri_api_concepts)
target_include_directories(psitri_api_concepts INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(psitri_api_concepts INTERFACE adb sal)
target_compile_features(psitri_api_concepts INTERFACE cxx_std_23)

add_library(psitri_api_small INTERFACE)
add_library(psitri::small ALIAS psitri_api_small)
target_link_libraries(psitri_api_small INTERFACE psitri_api_concepts start_ref)

add_library(psitri_api_big INTERFACE)
add_library(psitri::big ALIAS psitri_api_big)
target_link_libraries(psitri_api_big INTERFACE psitri_api_concepts sal_art_ref)

add_library(psitri_api INTERFACE)
add_library(psitri::api ALIAS psitri_api)
target_link_libraries(psitri_api INTERFACE psitri_api_small psitri_api_big)

# Compatibility targets for the initial API names.
add_library(psitri::fixed ALIAS psitri_api_small)
add_library(psitri::compact ALIAS psitri_api_big)
add_library(psitri_api_fixed ALIAS psitri_api_small)
add_library(psitri_api_compact ALIAS psitri_api_big)
