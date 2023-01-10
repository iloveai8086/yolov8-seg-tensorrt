#ifndef INFER_CONTROLLER_HPP
#define INFER_CONTROLLER_HPP

#include <string>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include "monopoly_allocator.hpp"

// 就是生产消费封装了
// 消费者的封装
template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int>
class InferController{
public:
    struct Job{
        Input input;
        Output output;
        JobAdditional additional;
        MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;
        std::shared_ptr<std::promise<Output>> pro;
    };

    virtual ~InferController(){
        stop();
    }

    void stop(){
        run_ = false;
        cond_.notify_all();

        ////////////////////////////////////////// cleanup jobs
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while(!jobs_.empty()){
                auto& item = jobs_.front();
                if(item.pro)
                    item.pro->set_value(Output());
                jobs_.pop();
            }
        };

        if(worker_){
            worker_->join();
            worker_.reset();
        }
    }

    bool startup(const StartParam& param){
        run_ = true;
        // promise，make_shared一个线程，然后传入成员函数worker，然后传入this和ref，典型的就是，消费者启动线程的一个流程
        // 其实就是为了，传递promise，在线程里面，在线程上下文内部做模型的加载，加载成功告诉我true，加载失败了告诉我false  set_value告诉的
        std::promise<bool> pro;
        start_param_ = param;  // 之前这边是好多的变量，也封装成一个类了
        worker_      = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));  // 消费者启动线程的流程
        // worker 线程其实是纯虚函数，实际一个子类线程，子类的函数
        // 线程内部做模型的加载，加载成功告诉true，否则告诉false
        // 是因为消费者有同样的共性，每次都得写同样的代码，不如封装起来。避免写重复的代码，避免每次都是什么notify one 创建promise返回future
        // worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro)); 之前是这样写的
        return pro.get_future().get();
    }

    virtual std::shared_future<Output> commit(const Input& input){

        Job job;
        job.pro = std::make_shared<std::promise<Output>>();
        if(!preprocess(job, input)){  // 预处理放到了cuda？
            job.pro->set_value(Output());
            return job.pro->get_future();
        }
        
        ///////////////////////////////////////////////////////////
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            jobs_.push(job);
        };
        cond_.notify_one();
        return job.pro->get_future();
    }

    virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs){

        int batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;
        for(int epoch = 0; epoch < nepoch; ++epoch){
            int begin = epoch * batch_size;
            int end   = std::min((int)inputs.size(), begin + batch_size);

            for(int i = begin; i < end; ++i){
                Job& job = jobs[i];
                job.pro = std::make_shared<std::promise<Output>>();
                if(!preprocess(job, inputs[i])){
                    job.pro->set_value(Output());
                }
                results[i] = job.pro->get_future();
            }

            ///////////////////////////////////////////////////////////
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                for(int i = begin; i < end; ++i){
                    jobs_.emplace(std::move(jobs[i]));
                };
            }
            cond_.notify_one();
        }
        return results;
    }

protected:
    virtual void worker(std::promise<bool>& result) = 0;
    virtual bool preprocess(Job& job, const Input& input) = 0;

    // 下面的这个函数就是封装了 condition 那个wait的操作流程
    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_jobs.clear();
        for(int i = 0; i < max_size && !jobs_.empty(); ++i){
            fetch_jobs.emplace_back(std::move(jobs_.front()));
            jobs_.pop();
        }
        return true;
    }

    virtual bool get_job_and_wait(Job& fetch_job){

        // 之前的worker里面手动写这一坨子的函数的，就是一批次的获取一个批次的
        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_job = std::move(jobs_.front());
        jobs_.pop();
        return true;
    }

protected:
    StartParam start_param_;
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;
};

#endif // INFER_CONTROLLER_HPP