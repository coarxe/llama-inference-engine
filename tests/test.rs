// use llama_cpp_2::{
//     context::{
//         LlamaContext,
//     },
//     model::{LlamaModel},
//     sampling::LlamaSampler,
// };

// fn assert_send<T: Send>() {}
// fn assert_sync<T: Sync>() {}

// #[test]
// fn llama_thread_safety() {
//     assert_send::<LlamaContext>();
//     assert_sync::<LlamaContext>();

//     assert_send::<LlamaModel>();
//     assert_sync::<LlamaModel>();

//     assert_send::<LlamaSampler>();
//     assert_sync::<LlamaSampler>();
// }

// use crate::scheduler::Node;
// use crate::schemas::{GenerateRequest, GenerationResult};

// fn assert_send<T: Send>() {}
// fn assert_sync<T: Sync>() {}

// fn test_types() {
//     assert_send::<Node>();
//     assert_sync::<Node>();

//     assert_send::<GenerateRequest>();
//     assert_send::<GenerationResult>();
// }