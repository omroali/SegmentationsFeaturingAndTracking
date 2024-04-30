# Matlab kalman filter implemetnation
# function [xp, Pp] = kalmanPredict(x, P, F, Q)
# % Prediction step of Kalman filter.
# % x: state vector
# % P: covariance matrix of x
# % F: matrix of motion model
# % Q: matrix of motion noise
# % Return predicted state vector xp and covariance Pp
# xp = F * x; % predict state
# Pp = F * P * F' + Q; % predict state covariance
# end

# function [xe, Pe] = kalmanUpdate(x, P, H, R, z)
# % Update step of Kalman filter.
# % x: state vector
# % P: covariance matrix of x
# % H: matrix of observation model
# % R: matrix of observation noise
# % z: observation vector
# % Return estimated state vector xe and covariance Pe
# S = H * P * H' + R; % innovation covariance
# K = P * H' * inv(S); % Kalman gain
# zp = H * x; % predicted observation
# %%%%%%%%% UNCOMMENT FOR VALIDATION GATING %%%%%%%%%%
# %gate = (z - zp)' * inv(S) * (z - zp);
# %if gate > 9.21
# % warning('Observation outside validation gate');
# % xe = x;
# % Pe = P;
# % return
# %end
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# xe = x + K * (z - zp); % estimated state
# Pe = P - K * S * K'; % estimated covariance
# end

# function [px, py] = kalmanTracking(z)
# % Track a target with a Kalman filter
# % z: observation vector
# % Return the estimated state position coordinates (px,py)
# dt = 0.033; % time interval
# N = length(z); % number of samples
# F = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]; % CV motion model
# Q = [0.01 0 0 0; 0 1 0 0; 0 0 0.01 0; 0 0 0 1]; % motion noise
# H = [1 0 0 0; 0 0 1 0]; % Cartesian observation model
# R = [4 0; 0 4]; % observation noise
# x = [0 0 0 0]'; % initial state
# P = Q; % initial state covariance
# s = zeros(4,N);
# for i = 1 : N
#  [xp, Pp] = kalmanPredict(x, P, F, Q);
#  [x, P] = kalmanUpdate(xp, Pp, H, R, z(:,i));
#  s(:,i) = x; % save current state
# end
# px = s(1,:); % NOTE: s(2, :) and s(4, :), not considered here,
# py = s(3,:); % contain the velocities on x and y respectively
# end


constant_velocity_motion_model = None
delta_t = 0.5


def kalman_filter(noise_coords: tuple):

    estimate_coords = noise_coords
    return estimate_coords



def root_mean_squared_error():
    # return std and mean
    pass


def root_mean_squared_to_ground_truth()
    # 
    pass

def main():
    pass


if __name__ == "__main__":
    main()
